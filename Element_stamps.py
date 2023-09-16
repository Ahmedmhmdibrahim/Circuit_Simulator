from typing import Dict,List
import numpy as np
Convert_unit_to_value = {'G': 1e9,
                         'M': 1e6,
                         'K': 1e3,
                         'm': 1e-3,
                         'u': 1e-6,
                         'n': 1e-9,
                         'p': 1e-12,
                         "f": 1e-15,
                         'nothing': 1
                         }


##################Essential elements####################
def res_stamp(Y : np.array , from_node : int  , to_node : int , res_value :int):

    Y[from_node,from_node] += 1 / res_value
    Y[to_node,to_node] += 1 / res_value
    Y[from_node,to_node] += -1 / res_value
    Y[to_node,from_node] += -1 / res_value
    return Y

def idc_stamp(J : np.array , from_node : int  , to_node : int , I_value :float):

    J[from_node] += -I_value
    J[to_node] += I_value
    return J

def vdc_stamp(Y : np.array, J : np.array ,from_node : int  , to_node : int , v_value :float , vdc_num : int):
        Y[from_node,vdc_num] = 1
        Y[to_node,vdc_num] = -1
        Y[vdc_num,to_node] = -1
        Y[vdc_num,from_node] = 1

        J[vdc_num] += v_value
        return Y, J
###non linear elements###


##################dependent sources#####################
def vccs_stamp(Y : np.array, from_nodes: tuple , to_nodes: tuple , gm : float):

    Y[from_nodes[0],from_nodes[1]] += gm
    Y[to_nodes[0],to_nodes[1]] += gm
    Y[from_nodes[0],to_nodes[1]] += -gm
    Y[to_nodes[0],from_nodes[1]] += -gm
    return Y

def vcvs_stamp(Y : np.array, from_nodes: tuple , to_nodes: tuple , A : float, vcvs_num : int):
    Y[vcvs_num,from_nodes[1]] += -A
    Y[vcvs_num,to_nodes[1]] += A
    Y[vcvs_num,from_nodes[0]] += 1
    Y[vcvs_num,to_nodes[0]] += -1

    Y[from_nodes[0],vcvs_num] += 1
    Y[to_nodes[0],vcvs_num] += -1
    Y[from_nodes[1],vcvs_num] += 0
    Y[to_nodes[1],vcvs_num] += 0
    return Y

def cccs_stamp(Y : np.array, from_nodes: tuple , to_nodes: tuple , A : float, cccs_num : int):

    Y[cccs_num,from_nodes[1]] += 1
    Y[cccs_num,to_nodes[1]] += -1
    Y[cccs_num,from_nodes[0]] += 0
    Y[cccs_num,to_nodes[0]] += 0

    Y[from_nodes[0],cccs_num] += A
    Y[to_nodes[0],cccs_num] += -A
    Y[from_nodes[1],cccs_num] += 1
    Y[to_nodes[1],cccs_num] += -1

    return Y

def ccvs_stamp(Y : np.array, from_nodes: tuple , to_nodes: tuple , Rm : float, ccvs_num : int):

    Y[ccvs_num,from_nodes[1]] += 1
    Y[ccvs_num,to_nodes[1]] += -1
    Y[ccvs_num,from_nodes[0]] += 0
    Y[ccvs_num,to_nodes[0]] += 0

    Y[from_nodes[0],ccvs_num] += 0
    Y[to_nodes[0],ccvs_num] += 0
    Y[from_nodes[1],ccvs_num] += 1
    Y[to_nodes[1],ccvs_num] += -1

    Y[ccvs_num+1][from_nodes[1]] += 0
    Y[ccvs_num+1][to_nodes[1]] += 0
    Y[ccvs_num+1][from_nodes[0]] += 1
    Y[ccvs_num+1][to_nodes[0]] += -1
    Y[ccvs_num+1][ccvs_num] += -Rm

    Y[from_nodes[0]][ccvs_num+1] += 1
    Y[to_nodes[0]][ccvs_num+1] += -1
    Y[from_nodes[1]][ccvs_num+1] += 0
    Y[to_nodes[1]][ccvs_num+1] += 0

    return Y

####################special elements#####################
def opamp_stamp(Y : np.array, neg_terminal: int , pos_terminal: int , out_terminal: int, opamp_num: int):

    Y[opamp_num,neg_terminal] += 1
    Y[opamp_num,pos_terminal] += -1
    Y[out_terminal,opamp_num] += 1

    return Y

