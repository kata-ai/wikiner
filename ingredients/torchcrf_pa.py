from typing import Optional

from torch.autograd import Variable
from torchcrf import CRF

class CRF_PA(CRF):
    def PA_forward(self, emissions: Variable,
                   tags: Variable,
                   mask: Optional[Variable] = None,
                   y_pa_batch: Optional[Variable] = None,
                   reduce: bool = True,):
        """Compute the log likelihood of the given sequence of tags and emission score.
        Arguments
        ---------
        emissions : :class:`~torch.autograd.Variable`
            Emission score tensor of size ``(seq_length, batch_size, num_tags)``.
        tags : :class:`~torch.autograd.Variable`
            Sequence of tags as ``LongTensor`` of size ``(seq_length, batch_size)``.
        mask : :class:`~torch.autograd.Variable`, optional
            Mask tensor as ``ByteTensor`` of size ``(seq_length, batch_size)``.
        y_pa_batch : :class:`~torch.autograd.Variable`, optional
            y_pa_batch tensor as ``ByteTensor`` of size ``(seq_length, batch_size)``.
        reduce : bool
            Whether to sum the log likelihood over the batch.
        Returns
        -------
        """
        pass

    @staticmethod
    def _pa_log_sum_exp(tensor: Variable, pre_real_tags: Variable, dim: int) -> Variable:
        pass
