import numpy as np
import torch
import pdb
import torch.nn as nn
from torch.autograd import Variable
# from pyquaternion import Quaternion

inds = np.array([0, -1, -2, -3, 1, 0, 3, -2, 2, -3, 0, 1, 3, 2, -1, 0]).reshape(4,4)


def hamilton_product(q1, q2):
    q_size = q1.size()
    # q1 = q1.view(-1, 4)
    # q2 = q2.view(-1, 4)
    q1_q2_prods = []
    for i in range(4):
        # Hack to make 0 as positive sign. add 0.01 to all the values..
        q2_permute_0 = q2[:, np.abs(inds[i][0])]
        q2_permute_0 = q2_permute_0 * np.sign(inds[i][0] + 0.01)

        q2_permute_1 = q2[:, np.abs(inds[i][1])]
        q2_permute_1 = q2_permute_1 * np.sign(inds[i][1] + 0.01)

        q2_permute_2 = q2[:, np.abs(inds[i][2])]
        q2_permute_2 = q2_permute_2 * np.sign(inds[i][2] + 0.01)

        q2_permute_3 = q2[:, np.abs(inds[i][3])]
        q2_permute_3 = q2_permute_3 * np.sign(inds[i][3] + 0.01)
        q2_permute = torch.stack([q2_permute_0, q2_permute_1, q2_permute_2, q2_permute_3], dim=1)
        q1q2_v1 = torch.sum(q1 * q2_permute, dim=1).view(-1, 1)
        q1_q2_prods.append(q1q2_v1)

    q_ham = torch.cat(q1_q2_prods, dim=1)
    # q_ham = q_ham.view(q_size)
    return q_ham


def rotate_quat(q1, q2):
    # q1 is N x 4
    # q2 is N x 4
    return hamilton_product(q1, q2)


def quat_conjugate(quat):
    # quat = quat.view(-1, 4)

    q0 = quat[:, 0]
    q1 = -1 * quat[:, 1]
    q2 = -1 * quat[:, 2]
    q3 = -1 * quat[:, 3]

    q_conj = torch.stack([q0, q1, q2, q3], dim=1)
    return q_conj


def get_random_quat():
    # q = Quaternion.random()
    q = (np.random.rand(4)*2 -1)
    q = q/np.linalg.norm(q)
    # q_n = np.array(q.elements, dtype=np.float32)
    quat = Variable(torch.from_numpy(q).float()).view(1, -1).view(1, -1)
    return quat, q

# def convert_quat_to_euler(quat):
#     q0 = Quaternion(quat.cpu().numpy())
#     return q0.degrees, q0.axis

def test_hamilton_product():
    conjugate_quat_module = quat_conjugate
    quat1, q1 = get_random_quat()
    quat1_c = conjugate_quat_module(quat1)
    quat1 = quat1.repeat(10, 1)
    quat1_c = quat1_c.repeat(10, 1)
    quat_product = hamilton_product(quat1, quat1_c)
    assert np.abs(1 - torch.mean(torch.sum(quat_product.view(-1, 4), 1)).item()) < 1E-4, 'Test1 error hamilton product'
    quat1, q1 = get_random_quat()
    quat2, q2 = get_random_quat()
    quat_product = hamilton_product(quat1, quat2).data.numpy().squeeze()
    import transformations
    q_product = transformations.quaternion_multiply(q1, q2)
    # q_product = np.array((q1 * q2).elements, dtype=np.float32)
    assert np.mean(np.abs(quat_product - q_product)) < 1E-4, 'Error in hamilton test 2'

if __name__ == "__main__":
    test_hamilton_product()
