import os
import torch
import logging
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)
writer = SummaryWriter()


class Metric():

    def __init__(self, args):
        self.args = args

    def get_lr(self, optimizer):
        return optimizer.state_dict()['param_groups'][0]['lr']

    def count_parameters(self, model): # 모델의 학습 가능한 매개변수 수를 계산
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    def cal_acc(self, yhat, y): # 예측된 레이블과 실제 레이블 사이의 정확도를 계산
        with torch.no_grad():
            yhat = yhat.max(dim=-1)[1]  # [0]: max value, [1]: index of max value
            acc = (yhat == y).float().mean()

        return acc

    def cal_time(self, start_time, end_time): # 경과 시간을 분과 초로 변환하여 반환
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

        return elapsed_mins, elapsed_secs

    def cal_dev_score(self, score, indicator): # 평가 중에 얻은 스코어를 계산하고 출력
        validation_score = score['score'] / score['iter']
        for key, value in indicator.items():
            indicator[key] /= score['iter']
            
        print("\n\nDomain-Clustering :\tRI: {:.4f}\tNMI: {:.4f}\tACC: {:.4f}\tPurity: {:.4f}".format(
            indicator['eval_RI'], indicator['eval_NMI'], indicator['eval_acc'], indicator['eval_purity']))
        
        print("\n\nSemantic-Relatedness :\teval_SR: {:.4f}}".format(
            indicator['eval_SR']))
        
        print("\n\nSession-Retrieval :\tMRR: {:.4f}\tMAP: {:.4f}".format(
            indicator['eval_MRR'], indicator['eval_MAP']))
        
        print("\n\nAlign-Uniformity :\tAlignment: {:.4f}\tAdjusted-Alignment: {:.4f}\tUniformity: {:.4f}".format(
            indicator['eval_alignment'], indicator['eval_adjusted_alignment'], indicator['eval_uniformity']))

        return validation_score

    def update_indicator(self, indicator, score): # 메트릭 인디케이터를 업데이트
        for key, value in indicator.items():
            if key == 'eval_RI':
                indicator[key] += score['eval_RI']
            elif key == 'eval_NMI':
                indicator[key] += score['eval_NMI']
            elif key == 'eval_acc':
                indicator[key] += score['eval_acc']
            elif key == 'eval_purity':
                indicator[key] += score['eval_purity']
            elif key == 'eval_SR':
                indicator[key] += score['eval_SR']
            elif key == 'eval_MRR':
                indicator[key] += score['eval_MRR']
            elif key == 'eval_MAP':
                indicator[key] += score['eval_MAP']
            elif key == 'eval_alignment':
                indicator[key] += score['eval_alignment']
            elif key == 'eval_adjusted_alignment':
                indicator[key] += score['eval_adjusted_alignment']
            elif key == 'eval_uniformity':
                indicator[key] += score['eval_uniformity']
            

    def draw_graph(self, cp): # Loss 및 Accuracy에 대한 그래프를 작성
        writer.add_scalars('loss_graph', {'train': cp['tl'], 'valid': cp['vl']}, cp['ep'])
        writer.add_scalars('acc_graph', {'train': cp['tma'], 'valid': cp['vma']}, cp['ep'])

    def performance_check(self, cp, config): # 에포크당 성능 출력
        print(f'\t==Epoch: {cp["ep"] + 1:02} | Epoch Time: {cp["epm"]}m {cp["eps"]}s==')
        print(f'\t==Train Loss: {cp["tl"]:.4f} | Train acc: {cp["tma"]:.4f}==')
        print(f'\t==Valid Loss: {cp["vl"]:.4f} | Valid acc: {cp["vma"]:.4f}==')
        print(f'\t==Epoch latest LR: {self.get_lr(config["optimizer"]):.9f}==\n')

    def print_size_of_model(self, model): # 모델의 크기를 계산하고 출력
        torch.save(model.state_dict(), "temp.p")
        print('Size (MB):', os.path.getsize("temp.p") / 1e6)
        os.remove('temp.p')

    def move2device(self, sample, device): # 샘플을 지정된 장치로 이동
        if len(sample) == 0:
            return {}

        def _move_to_device(maybe_tensor, device):
            if torch.is_tensor(maybe_tensor):
                return maybe_tensor.to(device)
            elif isinstance(maybe_tensor, dict):
                return {
                    key: _move_to_device(value, device)
                    for key, value in maybe_tensor.items()
                    }
            elif isinstance(maybe_tensor, list):
                return [_move_to_device(x, device) for x in maybe_tensor]
            elif isinstance(maybe_tensor, tuple):
                return [_move_to_device(x, device) for x in maybe_tensor]
            else:
                return maybe_tensor

        return _move_to_device(sample, device)

    def save_model(self, config, cp, pco):
        if not os.path.exists(config['args'].path_to_save):
            os.makedirs(config['args'].path_to_save)

        sorted_path = config['args'].path_to_save + config['args'].ckpt
        if cp['vs'] > pco['best_valid_score']:
            # pco['early_stop_patient'] = 0
            pco['best_valid_score'] = cp['vs']

            state = {'model': config['model'].state_dict(),
                     'optimizer': config['optimizer'].state_dict()}

            torch.save(state, sorted_path)
            print(f'\t## SAVE {sorted_path} |'
                  f' valid_score: {cp["vs"]:.4f} |'
                  f' epochs: {cp["ep"]} |'
                  f' steps: {cp["step"]} ##\n')

        # self.draw_graph(cp)
        # self.performance_check(cp, config)


def pytorch_cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    This function can be used as a faster replacement for 1-scipy.spatial.distance.cdist(a,b)
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    return torch.mm(a_norm, b_norm.transpose(0, 1))