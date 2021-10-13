import numpy as np
from colmap.scripts.python.read_write_model import read_model, qvec2rotmat
# from colmap.scripts.python.read_write_dense import read_array
from imageio import imread
import argparse
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
from time import time
import random
random.seed(0)
from tqdm import tqdm
import math
import copy

def get_image(idx):
    im = imread(src + '/dense/images/' + images[idx].name)

    q = images[idx].qvec
    R = qvec2rotmat(q)
    T = images[idx].tvec
    p = images[idx].xys
    pars = cameras[idx].params
    K = np.array([[pars[0], 0, pars[2]], [0, pars[1], pars[3]], [0, 0, 1]])
    pids = images[idx].point3D_ids

    return {
        'image': im,
        'K': K,
        'q': q,
        'R': R,
        'T': T,
        'xys': p,
        'ids': pids}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', dest='root', help='source dataset directory',
                        default='/mnt/nas/dataset_share/Image_Matching_Challange_Data', 
                        type=str)
    parser.add_argument('--pt_num', dest='num_p',
                        help='universal point number to be selected', 
                        default=50, type=int)
    parser.add_argument('--min_exist_num', dest='min_existence_num',
                        help='min num of img an anchor exists in', 
                        default=10, type=int)
    parser.add_argument('--dis_rate', dest='min_init_p_dis_th',
                        help='min distance rate when selecting points', 
                        default=1.0, type=float)
    parser.add_argument('--exist_dis_rate', dest='min_img_p_dis_th',
                        help='min distance rate when judging anchors\' existence', 
                        default=0.75, type=float)

    args = parser.parse_args()

    # num_p = 50
    # min_init_p_dis_th = 1.0
    # min_img_p_dis_th = 0.75
    # min_existence_num = 10
    # root = '/mnt/nas/dataset_share/Image_Matching_Challange_Data'
    seqs = ['brandenburg_gate', 'buckingham_palace', 'colosseum_exterior',
            'grand_place_brussels', 'hagia_sophia_interior', 'notre_dame_front_facade',
            'palace_of_westminster', 'pantheon_exterior', 'prague_old_town_square',
            'reichstag', 'sacre_coeur', 'st_peters_square', 'taj_mahal',
            'temple_nara_japan', 'trevi_fountain', 'westminster_abbey']


    for seq in seqs:
        src = args.root + '/' + seq
        zeropoints = []
        cameras, images, points = read_model(path=src + '/dense/sparse', ext='.bin')

        '''print(f'Cameras: {len(cameras)}')
        print(f'Images: {len(images)}')
        print(f'3D points: {len(points)}')'''

        indices = [i for i in cameras]

        '''set_all = set()
        for img_num in indices:
            set_all = set_all | set(images[img_num].point3D_ids)
        set_all.discard(-1)''' # set_all = points

        # xyz = []
        point_index = list(points.keys())
        # print(point_index)
        # rgb = []
        # for i in points:
        #     xyz.append(points[i].xyz)
        #     point_index.append(i)
        #     rgb.append(points[i].rgb)
        # xyz = np.array(xyz) # shape(x, 3)
        # rgb = np.array(rgb)
        point_index = np.array(point_index)

        point_fre = dict.fromkeys(point_index, 0)

        for index in indices:
            data = get_image(index)
            setp = set(images[index].point3D_ids)
            setp.discard(-1)
            for k in setp: 
                point_fre[k] += 1

        point_freq = copy.deepcopy(point_fre)

        for key in point_fre:
            if point_fre[key] < args.min_existence_num:
                point_freq.pop(key)

        xyzp = []

        for key in point_freq:
            xyzp.append(np.concatenate((points[key].xyz, [key,])))
        xyzp = np.array(xyzp)

        cur = []
        subset = []
        for p in data['ids']:
            if p >= 0:
                cur.append(np.concatenate((points[p].xyz,np.array([p,]))))
                subset.append(p)
        cur = np.array(cur) # shape-(num_of_points, 4) # 4:x,y,z,id
        
        '''
        allp = [points[i].xyz for i in points]
        allp = np.array(allp)
        #allp = allp / allp[:,2].reshape((len(allp[:,2]), 1))
        #fig = plt.figure(figsize=(12, 12))
        #ax = plt.subplot(111, projection='3d')
        #ax.scatter(-allp[:, 0], -allp[:, 2], -allp[:, 1])#'r.', markersize=3
        #-x:-15~5
        #-z:-30~10
        #-y:-4~2
        selected_point = []
        for item in allp:
            if -2 < item[0] < 15 and 0 < item[2] < 4 and -10 < item[1] < 20:
                selected_point.append(-1*item)
        selected_point = np.array(selected_point)

        fig = plt.figure(figsize=(12, 12))
        ax = plt.subplot(111, projection='3d')
        ax.scatter(selected_point[:, 0], selected_point[:, 2], selected_point[:, 1])
        #plt.axis('off')
        ax.set_zlabel('Z')
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
        #plt.xlim(-10, 10)
        #plt.ylim(-30, -20)
        #plt.zlim(-20, 0)
        plt.savefig('{}-3D-selected.jpg'.format(seq))
        plt.close()
        #fig = plt.figure(figsize=(12, 12))
        #plt.scatter(-allp[:, 2], -allp[:, 1])#'r.', markersize=3
        #plt.axis('off')
        #ax.set_zlabel('Z')
        #ax.set_ylabel('Y')
        #ax.set_xlabel('X')
        #plt.xlim(-10, 10)
        #plt.ylim(-30, -20)
        #plt.zlim(-20, 0)
        #plt.savefig('{}-3D-yz.jpg'.format(seq))
        #plt.close()
        print('fig done')
        '''

        l = len(point_freq)
        kps = [random.randint(0, l-1),]

        min_distance = np.sqrt(math.pow(np.max(xyzp[:,0]) - np.min(xyzp[:,0]), 2) +
                            math.pow(np.max(xyzp[:,1]) - np.min(xyzp[:,1]), 2) +
                            math.pow(np.max(xyzp[:,2]) - np.min(xyzp[:,2]), 2)) / args.num_p * args.min_init_p_dis_th
        # print('dis-{}'.format(min_distance))

        flag = 0
        dis = 0
        while len(kps) < args.num_p:
            flag = 0
            i = random.randint(0, l-1)
            for j in kps:
                dis = math.pow(xyzp[i, 0] - xyzp[j, 0], 2) + \
                    math.pow(xyzp[i, 1] - xyzp[j, 1], 2) + \
                    math.pow(xyzp[i, 2] - xyzp[j, 2], 2)
                if dis < min_distance**2:
                    flag = 1
                    break
            if flag == 0:
                kps.append(i)

        out_img_name = []
        total_num_points = []

        print('processing {}'.format(seq))
        for index in tqdm(indices):
            data = get_image(index)
            #depth = data['depth']
            K = data['K']
            R = data['R']
            T = data['T']
            setp = set(images[index].point3D_ids)
            setp.discard(-1)
            kpoints = []

            for p in kps:
                for img_p in setp:
                    dis = math.pow(xyzp[p, 0] - points[img_p].xyz[0], 2) + \
                        math.pow(xyzp[p, 1] - points[img_p].xyz[1], 2) + \
                        math.pow(xyzp[p, 2] - points[img_p].xyz[2], 2)
                    if dis < (min_distance * args.min_img_p_dis_th)**2:
                        kpoints.append(xyzp[p, :])
                        break

            # pt_id = 0
            # for p in kps:
            #     if point_index[p] in setp :
            #         kpoints.append(np.concatenate((xyz[p, :], np.array([pt_id,]))))
            #     pt_id += 1

            kpoints = np.array(kpoints)
            # print(kpoints.shape)

            if kpoints.shape[0] == 0:
                # fig = plt.figure(frameon=False) # figsize=(12, 12)
                # f = plt.imshow(data['image'])
                # plt.axis('off')
                zeropoints.append('{}-{}'.format(seq, images[index].name))

                output = []
                # output = (np.array(output)).T
                # np.savez('picture/{}/{}.npz'.format(seq, images[index].name), points = output)

                # plt.savefig('picture/{}/{}'.format(seq, images[index].name),
                #            bbox_inches='tight', pad_inches=0)
                # plt.close()

                # out_img_name.append(images[index].name)
                continue

            p_proj = np.dot(K, np.dot(R, kpoints[:, :3].T) + T[..., None])
            p_proj = p_proj / p_proj[2, :]

            check = []
            row, col, _ = data['image'].shape
            lenth = p_proj.shape[1]
            checkx = np.ones(lenth)
            checkx[p_proj[0] < 0] = 0
            checkx[p_proj[0] > col] = 0
            checky = np.ones(lenth)
            checky[p_proj[1] < 0] = 0
            checky[p_proj[1] > row] = 0
            checking = np.multiply(checkx,checky)
            for i in range(lenth):
                if checking[i] == 1:
                    check.append(p_proj[:, i])
            p_proj_in_img = (np.array(check)).T # shape(3,x) # x <= 50

            fig = plt.figure(frameon=False) # figsize=(12, 12)
            f = plt.imshow(data['image'])
            plt.axis('off')
            
            output = []

            if p_proj_in_img.shape[0] != 0:
                plt.plot(p_proj_in_img[0, :], p_proj_in_img[1, :], 'r.', markersize=3)
            else:
                zeropoints.append('{}-{}'.format(seq, images[index].name))
            for j in range(kpoints.shape[0]):
                if checking[j] == 1:
                    kp_id = int(np.where(kps == np.where(xyzp[:,3] == int(kpoints[j, 3]))[0])[0])
                    plt.text(p_proj[0, j], p_proj[1, j], '{}'.format(kp_id), 
                            horizontalalignment='center', verticalalignment='center',
                            size=8, color = 'orange')
                    output.append(np.array([kp_id, p_proj[0, j], p_proj[1, j]]))

            output = list(map(list, zip(*output)))
            if p_proj_in_img.shape[0] != 0:
                np.savez('picture/{}/{}.npz'.format(seq, images[index].name), points = output)

                plt.savefig('picture/{}/{}'.format(seq, images[index].name),
                            bbox_inches='tight', pad_inches=0)
            plt.close()

            if p_proj_in_img.shape[0] != 0:
                out_img_name.append(images[index].name)
                total_num_points.append(p_proj_in_img.shape[1])
        total_num_points = np.array(total_num_points)
        info = {'max': np.max(total_num_points), 'min': np.min(total_num_points),
                'average': np.average(total_num_points), 'var': np.var(total_num_points)}
        np.savez('picture/{}/img_info.npz'.format(seq), img_name = out_img_name, points_info = info)

    np.savez('picture/{}/zero_point_img.npz'.format(seq), img_name = zeropoints)
    # print(len(zeropoints))
    # print(zeropoints)
