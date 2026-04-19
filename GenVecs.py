import os
import torch
from tqdm import tqdm
from itertools import product

def gen_event_mats(orgin_dir, mat_dir, centertype, nodetype):
    if os.path.exists(mat_dir + 'Event_' + str(nodetype) + str(centertype) + '_a.pt'):
        return
    if os.path.exists(mat_dir + 'Event_' + str(nodetype) + str(centertype) + '_b.pt'):
        return
    files = os.listdir(mat_dir)
    for f in files:
        if 'Event_' in f and str(centertype) + '_b' in f and f.endswith('.pt'):
            filename = str(nodetype) + str(centertype)
            mat_a = torch.load(orgin_dir + filename + '.pt').cuda()
            for i in range(mat_a.shape[0]):
                deg = len(torch.where(mat_a[i] != 0)[0])
                mat_a[i] = torch.where(mat_a[i] != 0, deg, mat_a[i])
            mat_b = torch.load(mat_dir + f)
            torch.save(mat_a, mat_dir + 'Event_' + str(nodetype) + str(centertype) + '_a.pt')
            torch.save(mat_b, mat_dir + 'Event_' + str(nodetype) + str(centertype) + '_b.pt')
            return
    filename = str(nodetype) + str(centertype)
    mat_a = torch.load(orgin_dir + filename + '.pt').cuda()
    for i in range(mat_a.shape[0]):
        deg = len(torch.where(mat_a[i] != 0)[0])
        mat_a[i] = torch.where(mat_a[i] != 0, deg, mat_a[i])
    mat_b = torch.zeros((mat_a.shape[1], mat_a.shape[1]), device='cuda')
    for i in range(mat_b.shape[0]):
        mat_b[i][i] = len(torch.where(mat_a[:, i] != 0)[0])
    torch.save(mat_a, mat_dir + 'Event_' + str(nodetype) + str(centertype) + '_a.pt')
    torch.save(mat_b, mat_dir + 'Event_' + str(nodetype) + str(centertype) + '_b.pt')
    return 

def gen_edgeweight_mat(orgin_dir, mat_dir, centertype, nodetype):
    if os.path.exists(mat_dir + 'EdgeWeight_' + str(nodetype) + str(centertype) + '.pt'):
        return
    filename = str(nodetype) + str(centertype)
    edgeweight_mat = torch.load(orgin_dir + filename + '.pt').cuda()
    torch.save(edgeweight_mat, mat_dir + 'EdgeWeight_' + str(nodetype) + str(centertype) + '.pt')
    return

def genmats(dataset, centertype, nodetype):
    orgin_dir = 'Datasets/' + dataset + '/'
    mat_dir = 'Matrixs/' + dataset + '/'
    for i in range(len(nodetype)):
        if nodetype[i] == centertype:
            continue
        gen_event_mats(orgin_dir, mat_dir, centertype, nodetype[i])
        gen_edgeweight_mat(orgin_dir, mat_dir, centertype, nodetype[i])
    return

def get_E_dict(mat_dir, nodetypes, centertype):
    if os.path.exists(mat_dir + 'E.pt'):
        return
    unique_dict = {}
    E_dict = {}
    idx = 0
    if nodetypes[0] != centertype:
        col_count = torch.load(mat_dir + 'Event_' + str(nodetypes[0]) + str(centertype) + '_a.pt').shape[1]
    else:
        col_count = torch.load(mat_dir + 'Event_' + str(nodetypes[-1]) + str(centertype) + '_a.pt').shape[1]
    for col in tqdm(range(col_count)):
        for i in range(len(nodetypes)):
            if nodetypes[i] == centertype:
                continue
            unique_dict[str(nodetypes[i]) + str(centertype)] = []
            mat_a = torch.load(mat_dir + 'Event_' + str(nodetypes[i]) + str(centertype) + '_a.pt')[:, col]
            mat_w = torch.load(mat_dir + 'EdgeWeight_' + str(nodetypes[i]) + str(centertype) + '.pt')[:, col]
            num_b = torch.load(mat_dir + 'Event_' + str(nodetypes[i]) + str(centertype) + '_b.pt')[col][col]
            for j in range(mat_a.shape[0]):
                if mat_a[j] == 0:
                    continue
                num = int(str(int(mat_a[j].item())) + str(int(mat_w[j].item())) + str(int(num_b.item())))
                if num not in unique_dict[str(nodetypes[i]) + str(centertype)]:
                    unique_dict[str(nodetypes[i]) + str(centertype)].append(num)
        keys, values = zip(*unique_dict.items())
        unique_list = [''.join(map(str, v)) for v in product(*values)]
        for i in range(len(unique_list)):
            if unique_list[i] not in E_dict:
                E_dict[unique_list[i]] = idx
                idx += 1
    torch.save(E_dict, mat_dir + 'E.pt')
    return

def find_vec(mat_dir, nodetypes, centertype, row, col, val, name):
    event_dict = {}
    for j in range(len(nodetypes)):
        if nodetypes[j] == centertype:
            continue
        event_dict[str(nodetypes[j]) + str(centertype)] = []
        mat_a = torch.load(mat_dir + 'Event_' + str(nodetypes[j]) + str(centertype) + '_a.pt')[:, col]
        mat_w = torch.load(mat_dir + 'EdgeWeight_' + str(nodetypes[j]) + str(centertype) + '.pt')[:, col]
        num_b = torch.load(mat_dir + 'Event_' + str(nodetypes[j]) + str(centertype) + '_b.pt')[col][col]
        if name == str(nodetypes[j]) + str(centertype) + '_a.pt':
            num = int(str(val) + str(int(mat_w[row].item())) + str(int(num_b.item())))
            if num not in event_dict[str(nodetypes[j]) + str(centertype)]:
                event_dict[str(nodetypes[j]) + str(centertype)].append(num)
        else:
            for i in range(mat_a.shape[0]):
                if mat_a[i] == 0:
                    continue
                num = int(str(int(mat_a[i].item())) + str(int(mat_w[i].item())) + str(int(num_b.item())))
                if num not in event_dict[str(nodetypes[j]) + str(centertype)]:
                    event_dict[str(nodetypes[j]) + str(centertype)].append(num)
    keys, values = zip(*event_dict.items())
    event_list = [''.join(map(str, v)) for v in product(*values)]
    return event_list

def get_vecs(mat_dir, nodetype, nodetypes, centertype, aORb):
    if aORb != 'a' and aORb != 'b':
        raise ValueError("aORb must be 'a' or 'b'")
    if aORb == 'a' and os.path.exists(mat_dir + 'Vec_' + str(nodetype) + '.pt'):
        return
    if aORb == 'b' and os.path.exists(mat_dir + 'Vec_' + str(centertype) + '.pt'):
        return
    if aORb == 'a':
        print('Generating Vec_' + str(nodetype) + '.pt')
    if aORb == 'b':
        print('Generating Vec_' + str(centertype) + '.pt')
    E_dict = torch.load(mat_dir + 'E.pt')
    Event_mat = torch.load(mat_dir + 'Event_' + str(nodetype) + str(centertype) + '_' + aORb + '.pt')
    vec_mat = torch.zeros((Event_mat.shape[0], len(E_dict)), device='cuda')
    col_count = Event_mat.shape[1]
    for col in tqdm(range(col_count)):
        for j in range(vec_mat.shape[0]):
            if Event_mat[j][col] != 0:
                event_list = find_vec(mat_dir, nodetypes, centertype, j, col, int(Event_mat[j][col].item()), str(nodetype) + str(centertype) + '_' + aORb + '.pt')
                for k in range(len(event_list)):
                    vec_mat[j][E_dict[event_list[k]]] += 1
    if aORb == 'a':
        torch.save(vec_mat, mat_dir + 'Vec_' + str(nodetype) + '.pt')
    elif aORb == 'b':
        torch.save(vec_mat, mat_dir + 'Vec_' + str(centertype) + '.pt')
    return

def genvecs(dataset, centertype, nodetypes):
    mat_dir = 'Matrixs/' + dataset + '/'
    if not os.path.exists(mat_dir):
        os.makedirs(mat_dir)
    genmats(dataset, centertype, nodetypes)
    get_E_dict(mat_dir, nodetypes, centertype)
    for i in range(len(nodetypes)):
        if nodetypes[i] == centertype:
            continue
        get_vecs(mat_dir, nodetypes[i], nodetypes, centertype, 'a')
        get_vecs(mat_dir, nodetypes[i], nodetypes, centertype, 'b')
    return

if __name__ == '__main__':

    dataset = 'TEST'
    centertype = 1
    nodetypes = [0, 1, 2]
    genvecs(dataset, centertype, nodetypes)
