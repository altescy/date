from typing import Optional, cast

import torch
from allennlp.common.checks import check_dimensions_match
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder


@Seq2VecEncoder.register("satcked_seq2vec")
class StackedSeq2VecEncoder(Seq2VecEncoder):
    def __init__(
        self,
        seq2seq_encoder: Seq2SeqEncoder,
        seq2vec_encoder: Seq2VecEncoder,
        feedforward: FeedForward,
    ) -> None:
        check_dimensions_match(
            seq2seq_encoder.get_output_dim(),
            seq2vec_encoder.get_input_dim(),
            "seq2seq_encoder output dim",
            "seq2vec_encoder input dim",
        )
        check_dimensions_match(
            seq2vec_encoder.get_output_dim(),
            feedforward.get_input_dim(),  # type: ignore
            "seq2vec_encoder output dim",
            "feedforward input dim",
        )

        super().__init__()

        self._seq2seq_encoder = seq2seq_encoder
        self._seq2vec_encoder = seq2vec_encoder
        self._feedforward = feedforward

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        output = self._seq2seq_encoder(inputs, mask)
        output = self._seq2vec_encoder(output, mask)
        output = self._feedforward(output)
        return cast(torch.Tensor, output)

    def get_input_dim(self) -> int:
        return int(self._seq2seq_encoder.get_input_dim())

    def get_output_dim(self) -> int:
        return int(self._feedforward.get_output_dim())  # type: ignore
