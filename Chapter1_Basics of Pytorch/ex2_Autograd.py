import torch

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

BATCH_SIZE = 64     # 파라미터 업데이트시 계산되는 데이터의 개수
INPUT_SIZE = 1000   # Input의 크기이자 입력층의 노드 수
HIDDEN_SIZE = 100   # 은닉층의 노드 수
OUTPUT_SIZE = 10    # 최종 출력 벡터의 크기 (최종으로 비교하고자 하는 label의 크기와 동일하게)


# ===========Data와 Parameter설정===========

# Input Data ->
# Parameter Update를 위해 Gradient를 계산하는 것이지 Input에 Gradient 적용 X
x = torch.randn(BATCH_SIZE,
                INPUT_SIZE,
                device=DEVICE,
                dtype=torch.float,
                requires_grad=False)

# Output Data ->
# Input과 동일한 크기
y = torch.randn(BATCH_SIZE,
                OUTPUT_SIZE,
                device=DEVICE,
                dtype=torch.float,
                requires_grad=False)

# 업데이트할 파라미터 값 설정
w1 = torch.randn(INPUT_SIZE,
                HIDDEN_SIZE,
                device=DEVICE,
                dtype=torch.float,
                requires_grad=True)

# w1과 x를 곱한 결과에 계산할 수 있는 데이터 ->
# Back Propagation을 통해 업데이트 해야하는 대상
w2 = torch.randn(HIDDEN_SIZE,
                OUTPUT_SIZE,
                device=DEVICE,
                dtype=torch.float,
                requires_grad=True)

learning_rate = 1e-6                        # Gradient 값에 따른 학습 정도 결정
for t in range(1, 501):                     # 500회 반복하여 파라미터 업데이트
    y_pred = x.mm(w1).clamp(min=0).mm(w2)   # 예측값(딥러닝 결과값) 계산, clamp : 층과 층 사이에 비선형 함수 적용

    loss = (y_pred - y).pow(2).sum()        # 실제 값과의 비교 (오차)
    if t % 100 == 0:
        print("Iteration : ", t, "\t", "Loss : ", loss.item())
    loss.backward()                         # Gradient를 통한 Back Propagation

    with torch.no_grad():                   # 해당 시점의 Gradient 저장(고정)
        w1 -= learning_rate * w1.grad       # Loss 최소를 위한 Parameter Update ->
        w2 -= learning_rate * w2.grad       # Gradient값에 대한 반대방향

        w1.grad.zero_()                     # 파라미터 업데이트 이후 Gradient 초기화
        w2.grad.zero_()