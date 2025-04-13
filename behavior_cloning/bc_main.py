import pickle
import torch

def decompose(experiences):
    states = []
    actions = []
    for e in experiences:
        states.append(e[0])
        actions.append(e[1])
    return states, actions


def main():
    # 経験データを読み込み
    with open('experiences/experiences.pkl', 'rb') as f:
        experiences = pickle.load(f)
    states, actions = decompose(experiences)

    # T.B.D.

if __name__ == '__main__':
    main()

