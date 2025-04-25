from transformers import PreTrainedModel, RobertaConfig, RobertaModel
import torch
import torch.nn as nn

class RoBERTaSentenceConfig(RobertaConfig):
    model_type = "roberta_sentence"

    def __init__(self, num_labels=2, num_rnn_layers=1, num_fine_tuning_layers=3, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.num_rnn_layers = num_rnn_layers
        self.num_fine_tuning_layers = num_fine_tuning_layers

class RoBERTaSentence(PreTrainedModel):
    config_class = RoBERTaSentenceConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.roberta = RobertaModel.from_pretrained("roberta-base")

        roberta_hidden_size = self.roberta.config.hidden_size
        self.fine_tuning_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(roberta_hidden_size, roberta_hidden_size), nn.ReLU(), nn.Dropout(0.1))
            for _ in range(config.num_fine_tuning_layers)
        ])
        self.rnn = nn.GRU(roberta_hidden_size, 768, batch_first=True, num_layers=config.num_rnn_layers, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 128), nn.ReLU(), nn.Dropout(0.25), nn.Linear(128, config.num_labels)
        )
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def forward(self, input_ids, attention_mask, labels=None):
        roberta_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_states = roberta_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        for layer in self.fine_tuning_layers:
            last_hidden_states = layer(last_hidden_states)
        
        rnn_output, _ = self.rnn(last_hidden_states)  # (batch_size, seq_len, hidden_size * 2)
        cls_rnn_output = rnn_output[:, 0, :]  # (batch_size, hidden_size * 2)
        logits = self.classifier(cls_rnn_output)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return {"loss": loss, "logits": logits}

        return {"logits": logits}
