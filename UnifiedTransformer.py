import math
import torch
import torch.utils.checkpoint
from torch import nn
from typing import Dict
import json


class UnifiedTransformerConfig(Dict):
    def __init__(self):
        super(UnifiedTransformerConfig, self).__init__()

        self.vocab_size = 1000
        self.hidden_size = 1280
        self.activation_function = "gelu"
        self.decoder_filter_size = 5120
        self.decoder_attention_heads = 32
        self.decoder_layers = 12
        self.dropout = 0.0
        self.init_std = 0.02
        self.attention_dropout = 0.0
        self.activation_dropout = 0.0
        self.max_position_embeddings = 128
        self.type_token_embeddings = 2
        self.pad_token_id = 0
        self.scale_embedding = True
        self.attention_bias = True

    @classmethod
    def from_json_file(cls, file):

        config = UnifiedTransformerConfig()
        with open(file, "r") as f:
            config_dict = json.load(f)

        for key in list(config_dict.keys()):
            config.__dict__[key] = config_dict[key]

        return config

    def __str__(self):
        return str(self.__dict__)

class UnifiedTransformerAttention(nn.Module):
    def __init__(self, config:UnifiedTransformerConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.decoder_attention_heads
        self.dropout = config.attention_dropout
        self.head_dim = self.embed_dim // self.num_heads
        bias = config.attention_bias

        if (self.head_dim * self.num_heads) != self.embed_dim:
            assert f"embed_dim must be num_heads divisor, embed_dim:{self.embed_dim} num_heads:{self.num_heads}"
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)

    def transpose_shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states,
        key_value_states = None,
        attention_mask = None
    ):

        if key_value_states is None:
            key_states = hidden_states
            value_states = hidden_states
        else:
            key_states = key_value_states
            value_states = key_value_states

        bsz, tgt_len, _ = hidden_states.size()


        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self.transpose_shape(self.k_proj(key_states), -1, bsz)
        value_states = self.transpose_shape(self.v_proj(value_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self.transpose_shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                assert f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"

            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output


class UnifiedTransformerDecoderLayer(nn.Module):
    def __init__(self, config: UnifiedTransformerConfig):
        super().__init__()
        self.embed_dim = config.hidden_size

        self.self_attn = UnifiedTransformerAttention(
            config
        )
        self.dropout = config.dropout
        self.activation_fn = nn.GELU() if config.activation_function == "gelu" else nn.ReLU()
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = UnifiedTransformerAttention(
            config
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_filter_size)
        self.fc2 = nn.Linear(config.decoder_filter_size, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        encoder_hidden_states = None,
        encoder_attention_mask = None
    ):

        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            hidden_states = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        return hidden_states

class PretrainedModel(nn.Module):
    config = UnifiedTransformerConfig()
    def init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class UnifiedTransformerDecoder(PretrainedModel):

    def __init__(self, config: UnifiedTransformerConfig):
        super().__init__()

        self.embedding = Embeddings(config)

        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

        self.layers = nn.ModuleList([UnifiedTransformerDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        self.apply(self.init_weights)


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        type_token_ids=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None
    ):

        hidden_states = self.embedding(input_ids, position_ids=position_ids, type_token_ids=type_token_ids)

        for decoder_layer in self.layers:

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask
            )

        hidden_states = self.layer_norm(hidden_states)

        return hidden_states


class Embeddings(nn.Module):
    def __init__(self, config: UnifiedTransformerConfig):
        super(Embeddings, self).__init__()

        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=config.pad_token_id)
        self.type_token_embedding = nn.Embedding(config.type_token_embeddings, config.hidden_size, padding_idx=config.pad_token_id)

        self.positions = torch.arange(config.max_position_embeddings, dtype=torch.int32, device="cuda")

        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

        self.norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)


    def forward(self, input_ids=None, position_ids=None, type_token_ids=None):

        baz, seq_len = input_ids.shape

        input_emb = self.word_embedding(input_ids) * self.embed_scale

        if type_token_ids is not None:
            type_token_emb = self.type_token_embedding(type_token_ids)
            input_emb = input_emb + type_token_emb

        if position_ids is None:
            position_ids = self.positions[:seq_len]

        position_ids = position_ids.expand(baz, -1)
        position_emb = self.position_embedding(position_ids)
        input_emb = input_emb + position_emb

        embeddings = input_emb

        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class LMHead(nn.Module):

    def __init__(self, config, embedding=None):
        super(LMHead, self).__init__()
        self.transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation_fn = nn.GELU() if config.activation_function == "gelu" else nn.ReLU()
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.weight =  embedding.weight
        self.register_buffer("logits_bias", torch.zeros((1, embedding.num_embeddings)))

    def forward(self, hidden_states, masked_positions=None):
        if masked_positions is not None:

            hidden_states = hidden_states.view((-1, hidden_states.shape[-1]))
            hidden_states = hidden_states[masked_positions, :]
        hidden_states = self.transform(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = torch.matmul(hidden_states, self.weight.transpose(0, 1)) + self.logits_bias
        return logits


class UnifiedTransformerModel(PretrainedModel):

    def __init__(self, config: UnifiedTransformerConfig):
        super().__init__()

        self.decoder = UnifiedTransformerDecoder(config)
        self.lm_head = LMHead(config, self.decoder.embedding.word_embedding)
        self.apply(self.init_weights)

    def get_encoder(self):
        return self.encoder

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        type_token_ids = None,
        masked_positions = None
    ):


        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            type_token_ids=type_token_ids
        )

        lm_logits = self.lm_head(decoder_outputs, masked_positions)

        return lm_logits


if __name__ == '__main__':
    config = UnifiedTransformerConfig.from_json_file("config.json")
    model = UnifiedTransformerModel(config).to("cuda")

    inputs = {
        "input_ids": torch.ones((2, 24), dtype=torch.int32).to("cuda"),
        "attention_mask": torch.zeros((2, 1, 24, 24), dtype=torch.int32).to("cuda")
    }

    model(**inputs)
