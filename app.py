import gradio as gr
from miditok import REMI
from transformers import PretrainedConfig, PreTrainedModel
from reformer_pytorch import ReformerLM, Reformer
from axial_positional_embedding import AxialPositionalEmbedding
import math
import os
import subprocess
import pytube
import binascii

import torch
from torch import nn
import torchaudio

yt_dir = "./yt_dir"
midi_dir = "./midi_dir"
os.makedirs(yt_dir, exist_ok=True)
os.makedirs(midi_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# model define

class ReformerEncoderDecoderConfig(PretrainedConfig):
    def __init__(self,
                  vocab_size=50265, 
                  d_model=128,
                  num_heads=8, 
                  encoder_layers=6, 
                  decoder_layers=6, 
                  encoder_max_seq_len=6144,
                  decoder_max_seq_len=4096,
                  encoder_axial_position_shape=(96, 64),
                  decoder_axial_position_shape=(64, 64),
                  pad_token_id=0,
                  bos_token_id=1,
                  eos_token_id=2,
                  **kwargs):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_max_seq_len = encoder_max_seq_len
        self.decoder_max_seq_len = decoder_max_seq_len
        self.encoder_axial_position_shape = encoder_axial_position_shape
        self.decoder_axial_position_shape = decoder_axial_position_shape
        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id


class ReformerEncoderDecoder(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.pad_token_id = config.pad_token_id
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id

        self.encoder = Reformer(
            dim=config.d_model,
            depth=config.encoder_layers,
            heads=config.num_heads,
        )

        self.decoder = ReformerLM(
            dim=config.d_model,
            depth=config.decoder_layers,
            heads=config.num_heads,
            max_seq_len=config.decoder_max_seq_len,
            num_tokens=config.vocab_size,
            axial_position_emb=True,
            axial_position_shape=config.decoder_axial_position_shape,
            causal=True
        )

        self.position_embedding = AxialPositionalEmbedding(
            config.d_model,
            axial_shape=config.encoder_axial_position_shape
        )

    # https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/autopadder.py
    def pad_to_multiple(self, tensor, seq_len, multiple, dim=-1):
        m = seq_len / multiple
        if m.is_integer():
            return tensor
        
        remainder = math.ceil(m) * multiple - seq_len
        pad_offset = (0,) * (-1 - dim) * 2
        return nn.functional.pad(tensor, (*pad_offset, 0, remainder), value=self.pad_token_id)

    # https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/autopadder.py
    # pad_dim = -1 if its LM model else -2
    def auto_paddding(self, input_ids, pad_dim, bucket_size, num_mem_kv, full_attn_thres, keys=None, input_mask=None, input_attn_mask=None):
        device = input_ids.device

        batch_size, t = input_ids.shape[:2]

        keys_len = 0 if keys is None else keys.shape[1]
        seq_len = t + num_mem_kv + keys_len
        

        if seq_len > full_attn_thres:
            if input_mask is None:
                input_mask = torch.full((batch_size, t), True, dtype=torch.bool, device=device)

            input_ids = self.pad_to_multiple(input_ids, seq_len, bucket_size * 2, dim=pad_dim)

            if input_mask is not None:
                input_mask = nn.functional.pad(input_mask, (0, input_ids.shape[1] - input_mask.shape[1]), value=False)

            if input_attn_mask is not None:
                offset = input_ids.shape[1] - input_attn_mask.shape[1]
                input_attn_mask = nn.functional.pad(input_attn_mask, (0, offset, 0, offset), value=False)

        return input_ids, input_mask, input_attn_mask


    def shift_tokens_right(self, input_ids):
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = self.eos_token_id

        if self.pad_token_id is None:
            raise ValueError("config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, self.pad_token_id)

        return shifted_input_ids


    def forward(self, inputs_embeds, attention_mask=None, decoder_input=None, labels=None):
        if decoder_input is None:
            decoder_input = self.shift_tokens_right(labels)

        # encoder
        encoder_input = inputs_embeds + self.position_embedding(inputs_embeds)

        encoder_output = self.encoder(encoder_input, input_mask=attention_mask.bool())

        # decoder
        decoder_output = self.decoder(decoder_input, keys=encoder_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(decoder_output.view(-1, self.config.vocab_size), labels.view(-1))
            return {"loss": masked_lm_loss, "logits": decoder_output}
        
        return {"logits": decoder_output}


    @torch.no_grad()
    def generate(self, inputs_embeds, attention_mask=None, max_length=4096, temperature=1.0, top_k=50, top_p=1):
        is_training = self.training
        device = inputs_embeds.device

        # padding settings
        pad_dim = -1
        bucket_size = self.decoder.reformer.bucket_size
        num_mem_kv = self.decoder.reformer.num_mem_kv
        full_attn_thres = self.decoder.reformer.full_attn_thres

        self.eval()

        # encoder
        encoder_input = inputs_embeds + self.position_embedding(inputs_embeds)

        encoder_keys = self.encoder(encoder_input, input_mask=attention_mask.bool())

        # decoder
        generated = torch.tensor([self.bos_token_id], device=device).unsqueeze(0)

        decoder_mask = torch.full_like(generated, True, dtype=torch.bool, device=device)

        for _ in range(max_length):
            generated = generated[:, -self.config.decoder_max_seq_len:]
            decoder_mask = decoder_mask[:, -self.config.decoder_max_seq_len:]

            generated, decoder_mask, _ = self.auto_paddding(generated, 
                                                             pad_dim, 
                                                             bucket_size, 
                                                             num_mem_kv, 
                                                             full_attn_thres, 
                                                             keys=encoder_keys, 
                                                             input_mask=decoder_mask)
            
            logits = self.decoder(generated, input_mask=decoder_mask, keys=encoder_keys)[:, -1, :]  / temperature

            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(logits, top_k)
                filtered_logits = torch.full_like(logits, -float('Inf'))
                logits = filtered_logits.scatter(1, top_k_indices, top_k_values)

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0

                sorted_logits[sorted_indices_to_remove] = -float('Inf')
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)

            if next_token == self.eos_token_id:
                break

        self.train(is_training)
        return generated
    
# model define end

# model load
model = ReformerEncoderDecoder(ReformerEncoderDecoderConfig()).to(device)
model.load_state_dict(torch.load("model.pth"))
tokenizer = REMI(params="tokenizer.json")
# model load end
    

class ArrangerEmbedding(nn.Module):
  def __init__(self, arranger_ids=256, hidden_size=128):
    super().__init__()
    self.embeddings = nn.Embedding(arranger_ids, hidden_size)

  def forward(self, arranger_id, mel_db):
    return torch.cat([self.embeddings(arranger_id), mel_db], dim=-2)


def initialize_model(model_path):
    RedConfig = ReformerEncoderDecoderConfig()

    model = ReformerEncoderDecoder(RedConfig).cuda()

    model.load_state_dict(torch.load(model_path))

    return model


def load_input(song_path, arranger_id):
    waveform, sr = torchaudio.load(song_path)
    waveform = torchaudio.transforms.Resample(sr, 22050)(waveform)
    waveform = torch.mean(waveform, dim=0, keepdim=True)

    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=4096, hop_length=1024, n_mels=128)
    mel = mel_transform(waveform)
    mel_db = torchaudio.transforms.AmplitudeToDB()(mel)

    mel_shape = mel_db.shape
    mel_db = mel_db.reshape(mel_shape[0], mel_shape[2], mel_shape[1])

    if mel_db.shape[2] > 6144:
        mel_db = mel_db[:, :6144]

    num_pad = 6144 - mel_db.shape[1] - 1
    mel_padded = torch.cat([mel_db, torch.zeros((1, num_pad, mel_db.shape[2]))], dim=1)

    embbeding = ArrangerEmbedding()
    input_embed = embbeding(torch.tensor([[int(arranger_id)]]), mel_padded)
    attention_mask = torch.cat([torch.ones(mel_db.shape[:2], dtype=torch.int32), torch.zeros((mel_db.shape[0], num_pad + 1))], dim=1)

    return input_embed, attention_mask


def download_piano(youtube_link):
    yt = pytube.YouTube(youtube_link)
    download_path = os.path.join(yt_dir, yt.title + ".mp4")
    yt.streams.filter(only_audio=True).first().download(download_path)

    # convert to mp3
    mp3_path = str(download_path).replace(".mp4", ".mp3")
    result = subprocess.run([
        "ffmpeg",
        "-i", download_path,
        mp3_path
    ])

    if result.returncode != 0:
        raise Exception("Failed to convert to mp3")

    return mp3_path


def inference(yt_link, arranger_id):
    song_path = download_piano(yt_link)
    input_embed, attention_mask = load_input(song_path, arranger_id)
    generated = model.generate(input_embed.cuda(), attention_mask.cuda())
    return post_process(generated)


def post_process(generated):
    midi = tokenizer.decode(generated.argmax(dim=-1).cpu())

    # random name
    output_midi_path = os.path.join(midi_dir, f"{binascii.hexlify(os.urandom(8)).decode()}.mid")
    midi.dump_midi(os.path.join(midi_dir, output_midi_path))

    return output_midi_path


app = gr.Interface(
    fn=inference,
    inputs=[
        gr.Textbox(label="Youtube Link"),
        gr.Dropdown([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], label="Arranger ID", value=1)
    ],
    outputs=gr.File(label="MIDI File")
)

app.launch()
