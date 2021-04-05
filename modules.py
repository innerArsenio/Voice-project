from depen import *
from performer import SelfAttention, FeedForward, default
from attentions import AttentionLayer
#some copied from harvard

def swish(x):
    return x * torch.sigmoid(x)

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return swish(x)



def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)



class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1, d_change = None, ):
        super(PositionwiseFeedForward, self).__init__()
        d_last = d_model
        if d_change is not None:
            d_last = d_change
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_last)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(swish(self.w_1(x))))


class Softmax_att(nn.Module):
    def __init__(self, d_model, h, dropout):
        super(Softmax_att, self).__init__()
        self.att = nn.MultiheadAttention(d_model, h, dropout=dropout)
        print("softmax")

    def forward(self, x, context = None, mask = None, context_mask = None):

        context = default(context, x)
        context_mask = default(context_mask, mask)

        x = x.transpose(0,1).contiguous()
        context = context.transpose(0,1).contiguous()
        x, _= self.att(x, context, context, context_mask)
        
        return x.transpose(0,1).contiguous()


class Att_Choice(nn.Module):
    
    def __init__(self, type_att, d_model, h, dropout, local_window_size = 256, local_heads = 0, feature_redraw_interval = 1000, kernel_fn = nn.ReLU()):
        super(Att_Choice, self).__init__()
        self.type_att = type_att
        if type_att == "performer":
            self.att = SelfAttention(dim=d_model, heads=h, dropout=dropout, local_heads= local_heads,
                            local_window_size=local_window_size, feature_redraw_interval=feature_redraw_interval, kernel_fn=kernel_fn)
        elif type_att =="selfatt":
            self.att = Softmax_att(d_model, h, dropout=dropout)
        elif type_att == "linear":
            self.att = AttentionLayer(d_model, h)

    def forward(self, x, context = None, mask = None, context_mask = None):
        x = self.att(x, context, mask, context_mask)
        return x

def choose_ff(d_model, d_ff, dropout, type_c):
    print(type_c)
    if type_c == "classic":
        return PositionwiseFeedForward(d_model, d_ff, dropout)
    else:
        mult = int(d_ff/d_model)
        return FeedForward(d_model, mult, dropout, activation=Swish, glu = True)


class EncoderLayer(nn.Module):
    
    def __init__(self, attention_choice, ff_type, d_model, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.att = attention_choice
        self.ff = choose_ff(d_model, d_ff, dropout, ff_type)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        #B, N, D
        #N, B, D
        residual = x

        x = self.att(x, mask = mask)

        x = self.dropout1(x) + residual
        x = self.norm1(x)

        residual = x
        x = self.ff(x)
        x = self.dropout2(x) + residual
        x = self.norm2(x)
        return x

class Encoder(nn.Module):
    def __init__(self, embedding, position_encoder, layer, Number, norm = None):
        super(Encoder, self).__init__()
        self.embedding = embedding
        self.position_encoder = position_encoder
        self.layers = clones(layer, Number)
        self.norm = norm
    def forward(self, x, mask):
        if self.embedding is not None:
            x = self.embedding(x)
        x = self.position_encoder(x)
        for layer in self.layers:
            x = layer(x, mask)
        if self.norm is not None:
            x = self.norm(x)
        return x, mask



def make_encoder_text_model(hparams):
    model = Encoder(
        embedding=Embeddings(hparams.d_model_emb, hparams.vocab),
        position_encoder=PositionalEncoding(hparams.d_model_emb, hparams.dropout, hparams.pe_max_len),
        layer=EncoderLayer(Att_Choice(
                                        hparams.attention_type_text_encoder, 
                                        hparams.d_model_emb, hparams.heads, 
                                        hparams.dropout, int(hparams.local_window_size/2), hparams.local_heads,
                                        ), 
                            hparams.feedforward_type_text_encoder,
                            hparams.d_model_emb, hparams.d_ff, hparams.dropout),
        Number=hparams.encoder_number,
        norm = nn.LayerNorm(hparams.d_model_emb)
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def make_encoder_audio_model(hparams):
    model = Encoder(
        embedding=None,
        position_encoder=PositionalEncoding(hparams.n_mels, hparams.dropout, hparams.pe_mels_max_len),
        layer=EncoderLayer(Att_Choice(
                                        hparams.attention_type_audio_encoder, 
                                        hparams.n_mels, hparams.heads, 
                                        hparams.dropout, hparams.local_window_size, hparams.local_heads,
                                        ), 
                            hparams.feedforward_type_audio_encoder,
                            hparams.n_mels, hparams.n_mels_ff, hparams.dropout),        
        Number=hparams.encoder_number,
        norm = nn.LayerNorm(hparams.n_mels)
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class Decoder(nn.Module):
    def __init__(self, Size_changer, layer, Number, norm = None):
        super(Decoder, self).__init__()
        self.resize_module = Size_changer
        self.layers = clones(layer, Number)
        self.norm = norm
    def forward(self, text_input, text_mask, audio_input, audio_mask):
        text_input = self.resize_module(text_input)
        for layer in self.layers:
            audio_input = layer(text_input, text_mask, audio_input, audio_mask)
        if self.norm is not None:
            audio_input = self.norm(audio_input)
        return audio_input



class DecoderLayer(nn.Module):
    
    def __init__(self, attention_choice, ff_type, d_model, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.att = attention_choice
        self.ff = choose_ff(d_model, d_ff, dropout, ff_type)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, text_input, text_mask, audio_input, audio_mask):
        #B, N, D
        #N, B, D
        residual = audio_input
        
        x = self.att(audio_input, context = text_input, mask = audio_mask, context_mask = text_mask)
        
        x = self.dropout1(x) + residual
        x = self.norm1(x)

        residual = x
        x = self.ff(x)
        x = self.dropout2(x) + residual
        x = self.norm2(x)
        return x


def make_decoder_model(hparams):
    model = Decoder(
        Size_changer=PositionwiseFeedForward(hparams.d_model_emb, hparams.d_model_emb, dropout=hparams.dropout, d_change=hparams.n_mels),
        layer=DecoderLayer(Att_Choice(
                                        hparams.attention_type_decoder, 
                                        hparams.n_mels, hparams.heads, 
                                        hparams.dropout, 
                                        ), 
                            hparams.feedforward_type_decoder,
                            hparams.n_mels, hparams.n_mels_ff, hparams.dropout),   
        Number=hparams.decoder_number,
        norm= nn.LayerNorm(hparams.n_mels)
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class Model_Check(nn.Module):
    def __init__(self, hparams):
        super(Model_Check, self).__init__()
        self.text_decoder = make_encoder_text_model(hparams)
        self.audio_encoder = make_encoder_audio_model(hparams)
        self.decoder = make_decoder_model(hparams)
        self.ff_control = PositionwiseFeedForward(hparams.n_mels, hparams.n_mels)

    def forward(self, text_input, text_mask, audio_input, audio_mask):
        #([20, 1, 128, 1602]) -> 20, 1602, 128
        audio_input = audio_input.squeeze(1).transpose(1,2).contiguous()
        audio_input, audio_mask = self.audio_encoder.forward(audio_input, audio_mask)
        text_input, text_mask = self.text_decoder.forward(text_input, text_mask)
        x = self.decoder.forward(text_input, text_mask, audio_input, audio_mask)
        x = self.ff_control(x) + x
        return x
