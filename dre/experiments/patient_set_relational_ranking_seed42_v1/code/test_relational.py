"""Architecture, zero-residual and explicit context-use smoke checks."""
import torch

from relational import RelationalReadout, patient_equal_loss


def main():
    torch.manual_seed(42)
    z=torch.randn(8,64);m=torch.randn(8); y=torch.tensor([1.,0.,0.,1.,0.,1.,0.,1.])
    for name in ("R1_DEEPSET_RESIDUAL","R2_PAIRWISE_RELRANK","R3_WEIGHTED_RELRANK"):
        model=RelationalReadout(name)
        score=model(z,m,self_index=torch.arange(8))
        assert torch.equal(score,m),name
        loss,ce,rank=patient_equal_loss(score,y,.1)
        assert torch.isfinite(loss+ce+rank)
        opt=torch.optim.AdamW(model.parameters(),lr=.01)
        for _ in range(3):
            opt.zero_grad();patient_equal_loss(model(z,m,self_index=torch.arange(8)),y,.1)[0].backward();opt.step()
        a=model(z[:4],m[:4],z,m,torch.arange(4))
        b=model(z[:4],m[:4],z[:4],m[:4],torch.arange(4))
        assert torch.max(torch.abs(a-b))>1e-9,name
        a.sum().backward()
    print("RELATIONAL_UNIT_PASS")


if __name__=="__main__":main()
