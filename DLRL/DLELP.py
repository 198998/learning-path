# coding: utf-8
# Learning Path Recommendation via Dynamic Knowledge Graph Generation and Concept Similarity Analysis

import gym
import networkx as nx
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sys
import json
import pandas as pd
from dkt import DKT


sys.path.append('../')
from KT import Agent_KT
from AC import ActorCritic,Data_P
from PPO_Pre import PPO_Pre,Data_L
from PPO_Sim import PPO_Sim
from EduSim.Envs.KSS import KSSEnv, KSSAgent, kss_train_eval
from HGNN import hgnn
from dimkt import DIMKT
import ast




def train(env, Pre_agent, Sim_agent, Dif_agent, max_episode_num, batch_size, action_dim):

    th = 0.001
    difficult_levels = 10
    L_max_steps = 20


    df_concept_name = pd.read_csv('./data/selected_concept_name.csv')
    df_test = pd.read_csv('./data/test.csv')

    concept_df = pd.read_csv('./data/concept_id.csv', header=None, names=['concept', 'id', 'problem_id'], encoding='utf-8')
    concept_df['concept_lower'] = concept_df['concept'].str.lower()

    df = pd.read_csv('./data/knowledge_structure.csv')
    Know_G = nx.DiGraph()


    edges = list(zip(df['source'], df['target']))
    Know_G.add_edges_from(edges)

    with open('./data/concept_difficulty.json', 'r') as f:
        concept_difficulty = json.load(f)



    init_ques = df_test['concepts'].apply(lambda x: [int(i) for i in x.split(',')]).tolist()
    init_ans = df_test['responses'].apply(lambda x: [int(i) for i in x.split(',')]).tolist()

    init_diff = [
        [(concept_difficulty.get(str(concept), None) * difficult_levels) for concept in concept_list]
        for concept_list in init_ques
    ]

    lengths = [len(row) for row in init_ques]

    max_len = max(len(row) for row in init_ques)  # 找到最长的行长度

    # 用 0 填充较短的行，使得所有行长度相同
    init_ques = [row + [0] * (max_len - len(row)) for row in init_ques]
    init_ans = [row + [0] * (max_len - len(row)) for row in init_ans]
    init_diff = [row + [0] * (max_len - len(row)) for row in init_diff]

    init_h_q = init_ques
    init_h_a = init_ans
    init_h_diff = init_diff

    init_q = init_ques
    init_q = torch.tensor(init_q)
    init_q = init_q.unsqueeze(0).to(device)
    init_c = init_ques
    init_c = torch.tensor(init_c)
    init_c = init_c.unsqueeze(0).to(device)
    init_sd = init_diff
    init_sd = torch.tensor(init_sd)
    init_sd = init_sd.unsqueeze(0).to(device).long()
    init_qd = init_diff
    init_qd = torch.tensor(init_qd)
    init_qd = init_qd.unsqueeze(0).to(device).long()
    init_a = init_ans
    init_a = torch.tensor(init_a)
    init_a = init_a.unsqueeze(0).to(device)

    init_qshft = [[value + 1 for value in row] for row in init_ques]
    init_qshft = torch.tensor(init_qshft)
    init_qshft = init_qshft.unsqueeze(0).to(device)
    init_cshft = [[value + 1 for value in row] for row in init_ques]
    init_cshft = torch.tensor(init_cshft)
    init_cshft = init_cshft.unsqueeze(0).to(device)
    init_sdshft = [[value + 1 for value in row] for row in init_diff]
    init_sdshft = torch.tensor(init_sdshft)
    init_sdshft = init_sdshft.unsqueeze(0).to(device).long()
    init_qdshft = [[value + 1 for value in row] for row in init_diff]
    init_qdshft = torch.tensor(init_qdshft)
    init_qdshft = init_qdshft.unsqueeze(0).to(device).long()




    HGNN = hgnn(num_p=action_dim, num_c=action_dim, dp=action_dim/4, ddp=action_dim/4, dl=action_dim/4, ddl=action_dim/4, diff_lever=difficult_levels+2)

    dimkt = DIMKT(num_q=action_dim, num_c=action_dim, dropout=0.01, emb_size=action_dim, batch_size=40, num_steps=95,
                  difficult_levels=difficult_levels)


    student_num = len(df_concept_name)

    best_cal = 0


    for episode in tqdm.tqdm(range(max_episode_num),desc="Apisode"):

        env.reset()

        best_ep = -1
        count_ep = 0
        init_e = 0
        final_e = 0
        for index in tqdm.tqdm(range(student_num), desc="Student"):
            target = (concept_df.loc[
                          concept_df['concept'] == df_concept_name['selected_concept_name'][index], 'id'].values).astype(
                int).tolist()

            length = lengths[index]
            q = init_q[:, index, :]
            q = q[:, :length]
            c = init_c[:, index, :]
            c = c[:, :length]
            sd = init_sd[:, index, :]
            sd = sd[:, :length]
            qd = init_qd[:, index, :]
            qd = qd[:, :length]
            a = init_a[:, index, :]
            a = a[:, :length]

            qshft = init_qshft[:, index, :]
            qshft = qshft[:, :length]
            cshft = init_cshft[:, index, :]
            cshft = cshft[:, :length]

            sdshft = init_sdshft[:, index, :]
            sdshft = sdshft[:, :length]
            qdshft = init_qdshft[:, index, :]
            qdshft = qdshft[:, :length]

            init_ques = init_h_q[index][:length]
            init_diff = init_h_diff[index][:length]
            init_ans = init_h_a[index][:length]


            _, init_profile = env.begin_episode()



            KT = Agent_KT()

            state = KT.forward_state(init_ques,init_ans)
            state_n = state[:, -1, :]

            state_n1 = state_n.cpu().detach().numpy()

            if state_n1[0][target[0]] > 0.5: init_e += 1

            last_tor = 6 # demo
            last_prac_num = 5 # demo

            num_l = len(init_ques)



            q_embedding = HGNN(init_diff,init_ques,init_diff,init_ques)
            P_state = dimkt(q,c,sd,qd,a,qshft,cshft,sdshft,qdshft,q_embedding)

            next_P_state = P_state
            L_state = P_state
            pre_state =  L_state


            l_steps = 0
            l_knows = []
            know = None
            pre_action = 1
            isin = 0
            ifcontinue = 0

            lp = []
            while True:

                if pre_action == 1:
                    know, tolerance, _ = Pre_agent.take_action(L_state,know,target,Know_G,k_hop=1,threshold=0.6,last_tor=last_tor,last_prac_num=last_prac_num)
                    t_know = know

                if len(lp) > 0:
                    if (L_state[0][t_know] - pre_state[0][t_know]) <= th:
                        if pre_action == 1:
                            pre_action = 0
                            t_know = know
                            lp.pop()
                            l_knows.pop()
                        know, tolerance = Sim_agent.take_action(L_state, know, Know_G1, threshold=0.5,last_tor=last_tor,last_prac_num=last_prac_num)
                        if know == t_know:
                            if isin == 1:
                                isin = 0
                                pre_action = 1
                                ifcontinue = 1
                            else:
                                isin = 1
                    elif (L_state[0][t_know] - pre_state[0][t_know]) > th and pre_action == 0:
                        pre_action = 1
                        if isin == 1:
                            isin = 0
                            ifcontinue = 1
                        else:
                            know = t_know
                if ifcontinue == 1:
                    ifcontinue = 0
                    continue


                l_knows.append(know)
                p_steps = 0
                p_step_reward = 0

                if len(lp)== L_max_steps-1:
                    ques = target[0]
                else:

                    p_steps += 1

                    ques = Dif_agent.take_action(l_knows, P_state, init_diff)

                    next_P_state, observation, p_reward, done, init_ques1, init_ans1, q_diff = env.step_p(init_ques, init_ans, str(ques))


                    p_step_reward += p_reward
                    Dif_agent.store_transition(Data_P(P_state.cpu().detach().numpy(), int(ques), p_reward, next_P_state.cpu().detach().numpy(), done))

                    P_state = next_P_state

                pre_state = L_state
                lp.append(ques)
                if concept_difficulty.get(str(ques), None) is None:
                    next_diff = 0.5
                else:
                    next_diff = concept_difficulty.get(str(ques), None)
                next_diff = next_diff * difficult_levels
                init_diff.append(next_diff)
                init_ques.append(ques)
                init_ans.append(1 if state_n1[0][ques] > 0.5 else 0)
                q = torch.cat((q,torch.tensor([[ques]]).to(device)),dim=1)
                c = q
                sd  = torch.cat((sd,torch.tensor([[next_diff]]).to(device)),dim=1).long()
                qd = sd
                a = torch.cat((a,torch.tensor([[1 if state_n1[0][ques] > 0.5 else 0]]).to(device)),dim=1)
                if ques == 834:
                    qshft = torch.cat((qshft, torch.tensor([[ques]]).to(device)), dim=1)
                else:
                    qshft = torch.cat((qshft,torch.tensor([[ques+1]]).to(device)),dim=1)
                cshft = qshft
                sdshft = torch.cat((sdshft,torch.tensor([[next_diff+1]]).to(device)),dim=1).long()
                qdshft = sdshft


                q_embedding = HGNN(init_diff, init_ques, init_diff, init_ques)
                L_state = dimkt(q, c, sd, qd, a, qshft, cshft, sdshft, qdshft, q_embedding)

                if len(lp) >= L_max_steps:
                    final_state = KT.forward_state(init_ques, init_ans)
                    final_state = final_state[:, -1, :]
                    if final_state[0][target[0]] > 0.5: final_e += 1

                    difference = [(final_state[0][j] - state_n[0][j]) / (1 - state_n[0][j]) for j in target]
                    if len(difference) == 0: continue
                    diff_state = sum(difference) / len(difference)

                    count_ep = count_ep + diff_state
                    mean_ep = count_ep / (index + 1)

                    if diff_state > best_ep: best_ep = diff_state

                    print("Episode: {}, Mean: {}, Best: {}".format(index + 1, mean_ep, best_ep))
                    break






                if Dif_agent.memory_counter >= batch_size:
                    Dif_agent.learn()

                _, l_reward, _, _ = env.step_l()

                next_L_state = next_P_state
                Pre_agent.store_transition(Data_L(L_state.cpu().detach().numpy(), int(know), l_reward, next_L_state.cpu().detach().numpy(), done))
                Sim_agent.store_transition(Data_L(L_state.cpu().detach().numpy(), int(know), l_reward, next_L_state.cpu().detach().numpy(),done))

                last_tor = int(tolerance)
                last_prac_num = p_steps
        print("---calculate--- is", (final_e - init_e)/(student_num-init_e))
        if (final_e - init_e)/(student_num-init_e) > best_cal: best_cal = (final_e - init_e)/(student_num-init_e)
        print("---best calculate--- is", best_cal)
        if Pre_agent.memory_counter >= batch_size:
            Pre_agent.learn()
            Sim_agent.learn()
    return best_cal




if __name__ == '__main__':
    env = gym.make("KSS-v2", seed=10)
    state_dim = env.action_space.n
    action_dim = env.action_space.n
    hidden_dim = env.action_space.n
    actor_lr = 0.001
    critic_lr = 0.001
    gamma = 0.98

    lmbda = 0.95
    epochs = 10
    eps = 0.2


    max_episode_num = 100
    batch_size = 128


    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


    Pre_agent = PPO_Pre(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device,batch_size)

    Sim_agent = PPO_Sim(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device,batch_size)

    Dif_agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr,
                        critic_lr, gamma, env.learning_item_base.knowledge2item, device,batch_size)



    rewards = train(env, Pre_agent, Sim_agent, Dif_agent, max_episode_num, batch_size,action_dim)

    print(rewards)



