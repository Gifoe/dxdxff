"""No data/cache access; finite-gradient and patient-equal geometry checks."""
import torch

from geometry_losses import centroid_loss, cross_patient_supcon, direction_loss


def main():
    h = torch.tensor([[[1., 0.], [0., 1.], [0., 0.]],
                      [[.8, .2], [.2, .8], [0., 0.]]], requires_grad=True)
    y = torch.tensor([[1., 0., -1.], [1., 0., -1.]])
    mask = torch.tensor([[True, True, False], [True, True, False]])
    values = [direction_loss(h, y, mask), centroid_loss(h, y, mask),
              cross_patient_supcon(h, y, mask, .1)]
    assert all(torch.isfinite(v).item() for v in values)
    sum(values).backward()
    assert h.grad is not None and torch.isfinite(h.grad).all().item()
    assert not h.grad[:, 2].abs().any().item()
    assert direction_loss(h[:1], y[:1], mask[:1]).item() == 0
    assert centroid_loss(h[:1], y[:1], mask[:1]).item() == 0
    print("GEOMETRY_UNIT_PASS")


if __name__ == "__main__": main()
