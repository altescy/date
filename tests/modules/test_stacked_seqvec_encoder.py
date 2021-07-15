import torch
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2seq_encoders import LstmSeq2SeqEncoder
from allennlp.modules.seq2vec_encoders import LstmSeq2VecEncoder
from allennlp.nn.activations import SwishActivation

from date.modules.stacked_seq2vec_encoder import StackedSeq2VecEncoder


def test_stacked_seq2vec_encoder() -> None:
    module = StackedSeq2VecEncoder(
        LstmSeq2SeqEncoder(input_size=16, hidden_size=8),
        LstmSeq2VecEncoder(input_size=8, hidden_size=8),
        FeedForward(
            input_dim=8,
            num_layers=2,
            hidden_dims=4,
            activations=SwishActivation(),
        ),
    )

    inputs = torch.rand(4, 8, 16)
    output = module(inputs)

    assert output.size() == (4, 4)
