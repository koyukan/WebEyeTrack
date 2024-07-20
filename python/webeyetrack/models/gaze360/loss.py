
class PinBallLoss(nn.Module):
    def __init__(self):
        super(PinBallLoss, self).__init__()
        self.q1 = 0.1
        self.q9 = 1-self.q1

    def forward(self, output_o, target_o, var_o):
        q_10 = target_o-(output_o-var_o)
        q_90 = target_o-(output_o+var_o)

        loss_10 = torch.max(self.q1*q_10, (self.q1-1)*q_10)
        loss_90 = torch.max(self.q9*q_90, (self.q9-1)*q_90)

        loss_10 = torch.mean(loss_10)
        loss_90 = torch.mean(loss_90)

        return loss_10+loss_90