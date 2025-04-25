from transformers import PreTrainedModel, RobertaConfig, RobertaModel
import torch

class RoBERTaEnhancedFreezedConfig(RobertaConfig):
    model_type = "roberta_enhanced"
    def __init__(self, feature_dim=32, num_labels=2, num_rnn_layers=1, num_fine_tuning_layers=3, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.num_labels = num_labels
        self.num_rnn_layers = num_rnn_layers
        self.num_fine_tuning_layers = num_fine_tuning_layers

class RoBERTaEnhancedFreezed(PreTrainedModel):
    config_class = RoBERTaEnhancedFreezedConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.roberta = RobertaModel.from_pretrained("roberta-base")
        
        # Freeze all layers except the last 8 transformer layers
        for param in self.roberta.parameters():
            param.requires_grad = False
        for param in self.roberta.encoder.layer[-8:]:
            for p in param.parameters():
                p.requires_grad = True

        roberta_hidden_size = self.roberta.config.hidden_size
        combined_dim = roberta_hidden_size + config.feature_dim

        self.fine_tuning_layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(combined_dim, combined_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1)
            ) for _ in range(config.num_fine_tuning_layers)
        ])
        self.rnn = torch.nn.GRU(
            combined_dim,
            768,
            batch_first=True,
            num_layers=config.num_rnn_layers,
            bidirectional=True
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768 * 2, 128),  # Bi-directional output
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(128, config.num_labels)
        )
        self.post_init() 

    def _init_weights(self, module):
        """
        Initialize the weights of custom layers using Xavier initialization for linear layers
        and zero initialization for biases.
        """
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.GRU):
            for name, param in module.named_parameters():
                if "weight" in name:
                    torch.nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    torch.nn.init.zeros_(param)

    def forward(self, input_ids, attention_mask, features, labels=None):
        roberta_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_states = roberta_outputs.last_hidden_state
        cls_output = last_hidden_states[:, 0, :]  # CLS token output
        combined = torch.cat((cls_output, features), dim=1)

        # fine-tuning layers
        for layer in self.fine_tuning_layers:
            combined = layer(combined)

        # RNN
        rnn_output, _ = self.rnn(combined.unsqueeze(1))
        rnn_output = rnn_output.squeeze(1)
        #classifier
        logits = self.classifier(rnn_output)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return {"loss": loss, "logits": logits}

        return {"logits": logits}
