import torch
import torch.nn as nn
#the actual bread and butter of the model, very simply put
# Adding Resources https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
# FT treats every feature as a TOKEN , but we must specify and encode cont/cat featurs
# fed into encoder layers
# we can try activation='gelu' ingrid to keep reading
class FTTransformer(nn.Module):

    def __init__(self, cat_cardinalities, n_cont, embed_dim=32, n_layers=3, n_heads=4, n_classes=4):
        super().__init__()
        #tokenizer  Categorical data
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card + 1, embed_dim) for card in cat_cardinalities
        ])

        #ttokenizer  continuous data
        self.cont_weights = nn.Parameter(torch.empty(n_cont, embed_dim))
        self.cont_biases = nn.Parameter(torch.empty(n_cont, embed_dim))
        nn.init.kaiming_uniform_(self.cont_weights)
        nn.init.zeros_(self.cont_biases)

        # transformer Backbone + matching archiecure of the FT- transformer
        # setting batch_Firs makes the input shape (batch, seq, feaure) soooo (batch)size, num)freaturesm embed_dim)
        # sequence lengh is actaly number of columns FEATURES

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # classifier Head
        self.mlp = nn.Sequential(
            nn.Linear((len(cat_cardinalities) + n_cont) * embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )
#processes both cat and cont features, maps to embedded space and passed throgh transformer backbone to learn interaciotns between feaures
    #
    def forward(self, x_cat, x_cont):
        #iteratives throgh list of predefined embedding layers one for each categorical column
        #converting eachinto a dense vector
        #cat has shape [batch_Size, num_categorical_feaures, embedding_dim] each feaure is a dense vector
        cat_x = torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)], dim=1)
        #applyig ransormation and change of shape
        cont_x = x_cont.unsqueeze(-1) * self.cont_weights + self.cont_biases
        #combining embedeed categorical features and cont features into a single sequence
        x = torch.cat([cat_x, cont_x], dim=1)
        #passsing htrough encoder layer
        x = self.transformer(x)
        #returning flattened tensor producing final predicion
        return self.mlp(x.flatten(1))