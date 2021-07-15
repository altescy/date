import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
from allennlp.common.checks import check_dimensions_match
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.nn import InitializerApplicator, util
from allennlp.training.metrics import Auc

from date.mask import decode_mask


@Model.register("date")
class DATE(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        anomaly_label: str,
        text_field_embedder: TextFieldEmbedder,
        rmd_seq2vec_encoder: Seq2VecEncoder,
        rtd_seq2seq_encoder: Seq2SeqEncoder,
        masks: List[str],
        mlm_weight: float = 1.0,
        rmd_weight: float = 100.0,
        rtd_weight: float = 50.0,
        dropout: float = 0.0,
        mask_token: str = "[MASK]",
        cls_token: str = "[CLS]",
        namespace: str = "tags",
        label_namespace: str = "labels",
        primary_token_indexer: str = "bert",
        initializer: Optional[InitializerApplicator] = None,
        **kwargs: Any,
    ) -> None:
        check_dimensions_match(
            text_field_embedder.get_output_dim(),
            rmd_seq2vec_encoder.get_input_dim(),
            "text_field_embedder output dim",
            "rmd_seq2vec_encoder input dim",
        )
        check_dimensions_match(
            text_field_embedder.get_output_dim(),
            rtd_seq2seq_encoder.get_input_dim(),
            "text_field_embedder output dim",
            "rtd_seq2seq_encoder input dim",
        )

        super().__init__(vocab, **kwargs)

        self._anomaly_label = anomaly_label
        self._text_field_embbedder = text_field_embedder
        self._rmd_seq2vec_encoder = rmd_seq2vec_encoder
        self._rtd_seq2seq_encoder = rtd_seq2seq_encoder

        self._masks = [decode_mask(mask) for mask in masks]

        self._generator_projection = TimeDistributed(  # type: ignore
            torch.nn.Linear(
                text_field_embedder.get_output_dim(),
                self.vocab.get_vocab_size(namespace),
            )
        )
        self._rmd_projection = torch.nn.Linear(
            rmd_seq2vec_encoder.get_output_dim(), len(masks)
        )
        self._rtd_projection = TimeDistributed(  # type: ignore
            torch.nn.Linear(rtd_seq2seq_encoder.get_output_dim(), 2)
        )

        self._mlm_weight = mlm_weight
        self._rmd_weight = rmd_weight
        self._rtd_weight = rtd_weight

        self._mask_token_index = self.vocab.get_token_index(mask_token, namespace)
        self._cls_token_index = self.vocab.get_token_index(cls_token, namespace)
        self._namespace = namespace
        self._label_namespace = label_namespace
        self._primary_token_indexer = primary_token_indexer
        self._anomaly_label = anomaly_label
        self._anomaly_label_index = self.vocab.get_token_index(
            anomaly_label,
            label_namespace,
        )

        self._rmd_loss = torch.nn.CrossEntropyLoss()
        self._rtd_loss = torch.nn.CrossEntropyLoss()
        self._auc = Auc()  # type: ignore

        initializer = initializer or InitializerApplicator()
        initializer(self)

    def forward(  # type: ignore[override]
        self,
        tokens: TextFieldTensors,
        label: Optional[torch.LongTensor] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, max_tokens)
        mask = util.get_text_field_mask(tokens)
        # Shape: (batch_size, max_tokens)
        token_ids = cast(
            torch.LongTensor,
            util.get_token_ids_from_text_field_tensors(tokens),
        )

        output_dict: Dict[str, torch.Tensor] = {}
        output_dict["token_ids"] = token_ids
        output_dict["mlm_loss"] = torch.tensor(0.0)
        output_dict["rmd_loss"] = torch.tensor(0.0)
        output_dict["rtd_loss"] = torch.tensor(0.0)

        if self.training:
            # Shape: (batch_size, max_tokens)
            # Shape: (batch_size, )
            # Shape: (batch_size, max_tokens)
            masked_tokens, mask_labels, replaced_mask = self._mask_tokens(tokens)

            # Shape: (batch_size, max_tokens, embedding_dim)
            generator_embeddings = self._text_field_embbedder(masked_tokens)

            # Shape: (batch_size, max_tokens, num_vocab)
            vocab_logits = self._generator_projection(generator_embeddings)
            tokens = self._replace_masked_tokens(masked_tokens, vocab_logits)

            mlm_loss = util.sequence_cross_entropy_with_logits(
                vocab_logits, token_ids, replaced_mask
            )
            output_dict["mlm_loss"] = mlm_loss

        # Shape: (batch_size, max_tokens, embedding_dim)
        discriminator_embeddings = self._text_field_embbedder(tokens)

        if self.training:
            # Shape: (batch_size, encoding_dim)
            rmd_vectors = self._rmd_seq2vec_encoder(discriminator_embeddings, mask)
            # Shape: (batch_size, num_masks)
            rmd_logits = self._rmd_projection(rmd_vectors)
            if self.training:
                output_dict["rmd_loss"] = self._rmd_loss(rmd_logits, mask_labels)

        # Shape: (batch_size, max_tokens, 2)
        rtd_logits = self._rtd_projection(discriminator_embeddings)
        if self.training:
            output_dict["rtd_loss"] = util.sequence_cross_entropy_with_logits(
                rtd_logits,
                cast(torch.LongTensor, replaced_mask.long()),
                cast(torch.BoolTensor, mask & (token_ids != self._cls_token_index)),
            )

        output_dict["loss"] = (
            self._mlm_weight * output_dict["mlm_loss"]
            + self._rmd_weight * output_dict["rmd_loss"]
            + self._rtd_weight * output_dict["rtd_loss"]
        )

        # Shape: (batch_size, max_tokens)
        token_anomaly_scores = rtd_logits.softmax(-1)[:, :, 1]
        # Shape: (batch_size, )
        anomaly_score = token_anomaly_scores.mean(1)

        output_dict["token_anomaly_scores"] = token_anomaly_scores
        output_dict["anomaly_score"] = anomaly_score

        if label is not None:
            binary_label = label == self._anomaly_label_index
            self._auc(anomaly_score, binary_label)

        return output_dict

    def _mask_tokens(
        self,
        tokens: TextFieldTensors,
    ) -> Tuple[TextFieldTensors, torch.LongTensor, torch.BoolTensor]:
        # Shape: (batch_size, max_tokens)
        text_field_mask = util.get_text_field_mask(tokens)

        batch_size, max_tokens = text_field_mask.size()
        num_masks = len(self._masks)

        mask_ids = torch.LongTensor(
            [random.randrange(num_masks) for _ in range(batch_size)]
        )
        replaced_mask = torch.zeros((batch_size, max_tokens)).bool()
        for idx, mask_id in enumerate(mask_ids):
            _mask = self._masks[mask_id]
            length = min(max_tokens, len(_mask))
            replaced_mask[idx][:length] = torch.BoolTensor(_mask[:length])

        # Shape: (batch_size * num_masks, max_tokens)
        replaced_mask = cast(torch.BoolTensor, replaced_mask & text_field_mask)

        masked_tokens: TextFieldTensors = defaultdict(dict)
        for indexer_name, args in tokens.items():
            for arg_name, value in args.items():
                if arg_name in ("tokens", "token_ids", "input_ids"):
                    token_ids = value
                    token_ids = util.replace_masked_values(
                        token_ids,
                        cast(torch.BoolTensor, ~replaced_mask),
                        self._mask_token_index,
                    )
                    masked_tokens[indexer_name][arg_name] = token_ids
                else:
                    masked_tokens[indexer_name][arg_name] = value

        return masked_tokens, torch.LongTensor(mask_ids), replaced_mask

    def _replace_masked_tokens(
        self,
        tokens: TextFieldTensors,
        vocab_logits: torch.Tensor,
    ) -> TextFieldTensors:
        cd = torch.distributions.Categorical(logits=vocab_logits)  # type: ignore
        # Shape: (batch_size, max_tokens)
        sampled_token_ids = cd.sample()  # type: ignore

        replaced_tokens: TextFieldTensors = defaultdict(dict)
        indexer_name = self._primary_token_indexer
        for arg_name, value in tokens[indexer_name].items():
            if arg_name in ("tokens", "token_ids", "input_ids"):
                token_ids = value
                mask = token_ids != self._mask_token_index
                token_ids = torch.where(mask, token_ids, sampled_token_ids)
                replaced_tokens[indexer_name][arg_name] = token_ids
            else:
                replaced_tokens[indexer_name][arg_name] = value

        return replaced_tokens

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        tokens: List[List[str]] = []
        if "token_ids" in output_dict:
            for instance_tokens in output_dict["token_ids"]:
                tokens.append(
                    [
                        self.vocab.get_token_from_index(
                            token_id.item(), namespace=self._namespace
                        )
                        for token_id in instance_tokens
                    ]
                )
            output_dict["tokens"] = tokens  # type: ignore
            del output_dict["token_ids"]
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"auc": self._auc.get_metric(reset)}
        return metrics
