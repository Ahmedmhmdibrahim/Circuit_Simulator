import numpy as np
import Element_stamps
from typing import List, Dict
import matplotlib.pyplot as plt
from sympy import symbols
from sympy.matrices import Matrix

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

def matrix_formulation_OP(elements):
    n = elements["num_nets"] + elements["vsource_list"].__len__() + elements["inductor_list"].__len__() + elements["vcvs_list"].__len__() \
        + elements["cccs_list"].__len__() + 2*elements["ccvs_list"].__len__() + elements["opamp_list"].__len__()
    Y = np.zeros([n+1, n+1])
    J = np.zeros([n+1, 1])
    V = []
    # Construct Unknown Vector 'V'
    for i in range(elements["num_nets"]):
        V.append(f"V{i+1}")


    for element in elements["resistor_list"]:
        from_node = element["from"]
        to_node = element["to"]
        value = element["value"] * Convert_unit_to_value[element["unit"]]
        Element_stamps.res_stamp(Y, from_node = from_node , to_node= to_node , res_value= value)

    for element in elements["isource_list"]:
        from_node = element["from"]
        to_node = element["to"]
        value = element["value"] * Convert_unit_to_value[element["unit"]]
        Element_stamps.idc_stamp(J, from_node = from_node , to_node= to_node , I_value= value)

    for element in elements["capacitor_list"]:
        # o.c is an Isource with I = 0
        from_node = element["from"]
        to_node = element["to"]
        value = 0
        Element_stamps.idc_stamp(J, from_node = from_node , to_node= to_node , I_value= value)

    position =  elements["num_nets"]
    for i, element in enumerate(elements["vsource_list"]):
        from_node = element["from"]
        to_node = element["to"]
        vdc_num = position + i + 1
        if element["type"] == "dc":
            v_value = element["value"] * Convert_unit_to_value[element["unit"]]
        else:
            v_value = 0
        V.append("I_" + element["instance_name"])
        Element_stamps.vdc_stamp(Y, J, from_node = from_node , to_node= to_node , v_value= v_value , vdc_num = vdc_num)

    position += elements["vsource_list"].__len__()
    for i, element in enumerate(elements["inductor_list"]):
        # s.c is a Vsource with E = 0
        from_node = element["from"]
        to_node = element["to"]
        ind_num = position + i + 1
        V.append("I_" + element["instance_name"])
        Element_stamps.vdc_stamp(Y, J, from_node = from_node , to_node= to_node , v_value= 0 , vdc_num = ind_num)

    
    for i, element in enumerate(elements["vccs_list"]):
        from_node_1 = element["from_1"]
        to_node_1 = element["to_1"]
        from_node_2 = element["from_2"]
        to_node_2 = element["to_2"]
        gm = element["value"] * Convert_unit_to_value[element["unit"]]
        Element_stamps.vccs_stamp(Y, from_nodes = (from_node_1,from_node_2) , to_nodes= (to_node_1,to_node_2) , gm= gm)

    position += elements["inductor_list"].__len__()
    for i, element in enumerate(elements["vcvs_list"]):
        from_node_1 = element["from_1"]
        to_node_1 = element["to_1"]
        from_node_2 = element["from_2"]
        to_node_2 = element["to_2"]
        vcvs_num = position + i + 1
        A = element["value"] * Convert_unit_to_value[element["unit"]]
        V.append("I_" + element["instance_name"])
        Element_stamps.vcvs_stamp(Y, from_nodes = (from_node_1,from_node_2) , to_nodes= (to_node_1,to_node_2) , A=A, vcvs_num = vcvs_num)

    position += elements["vcvs_list"].__len__()
    for i, element in enumerate(elements["cccs_list"]):
        from_node_1 = element["from_1"]
        to_node_1 = element["to_1"]
        from_node_2 = element["from_2"]
        to_node_2 = element["to_2"]
        cccs_num = position + i + 1
        A = element["value"] * Convert_unit_to_value[element["unit"]]
        V.append("I_" + element["instance_name"])

        Element_stamps.cccs_stamp(Y, from_nodes = (from_node_1,from_node_2) , to_nodes= (to_node_1,to_node_2) , A=A, cccs_num = cccs_num)

    position += elements["cccs_list"].__len__()
    for i, element in enumerate(elements["ccvs_list"]):
        from_node_1 = element["from_1"]
        to_node_1 = element["to_1"]
        from_node_2 = element["from_2"]
        to_node_2 = element["to_2"]
        ccvs_num = position + i + 1
        Rm = element["value"] * Convert_unit_to_value[element["unit"]]

        V.append("I_" + element["instance_name"] + "_1")
        V.append("I_" + element["instance_name"] + "_2")
        Element_stamps.ccvs_stamp(Y, from_nodes = (from_node_1,from_node_2) , to_nodes= (to_node_1,to_node_2) , Rm=Rm, ccvs_num = ccvs_num)

    position += elements["ccvs_list"].__len__()
    for i, element in enumerate(elements["opamp_list"]):
        neg_terminal = element["neg_terminal"]
        pos_terminal = element["pos_terminal"]
        out_terminal = element["out_terminal"]
        opamp_num = position + i + 1

        V.append("I_" + element["instance_name"])
        Element_stamps.opamp_stamp(Y, neg_terminal = neg_terminal , pos_terminal = pos_terminal , out_terminal = out_terminal, opamp_num = opamp_num)

    # remove the zeroth line
    Y = Y[1:len(Y), 1:len(Y)]
    J = J[1:len(J)]
    return Y, V, J

def matrix_formulation_OP_non_linear(elements, last_iter_results):
    n = elements["num_nets"] \
        + elements["vsource_list"].__len__() \
        + elements["inductor_list"].__len__() \
        + elements["vcvs_list"].__len__() \
        + elements["cccs_list"].__len__() \
        + 2 * elements["ccvs_list"].__len__() \
        + elements["opamp_list"].__len__()
    j_f = np.zeros([n + 1, n + 1])
    s_k = np.zeros([n + 1, 1])
    V = []

    # add the volt of the ground
    last_iter_results = np.vstack((np.array([0]), last_iter_results))
    # Construct Unknown Vector 'V'
    for i in range(elements["num_nets"]):
        V.append(f"V{i + 1}")       #add the voltages of the nets, the currents is added later in each element needs it

    for element in elements["resistor_list"]:
        from_node = element["from"]
        to_node = element["to"]
        value = element["value"] * Convert_unit_to_value[element["unit"]]
        Element_stamps.res_stamp(j_f, from_node=from_node, to_node=to_node, res_value=value)

    for element in elements["isource_list"]:
        from_node = element["from"]
        to_node = element["to"]
        value = element["value"] * Convert_unit_to_value[element["unit"]]
        Element_stamps.idc_stamp(s_k, from_node=from_node, to_node=to_node, I_value=value)

    for element in elements["capacitor_list"]:
        # o.c is an Isource with I = 0
        from_node = element["from"]
        to_node = element["to"]
        value = 0
        Element_stamps.idc_stamp(j_f, from_node=from_node, to_node=to_node, I_value=value)

    for i, element in enumerate(elements["diode_list"]):
        from_node = element["from"]
        to_node = element["to"]
        temp = 27
        I_s = 1e-13
        V_T = 25.8e-3

        volt_drop = float(last_iter_results[from_node] - last_iter_results[to_node])
        G_eq = (I_s/V_T)*np.exp(volt_drop/V_T)
        I_eq = I_s*(np.exp(volt_drop/V_T) - 1) - volt_drop*G_eq

        Element_stamps.res_stamp(j_f, from_node=from_node, to_node=to_node, res_value=1/G_eq)
        Element_stamps.idc_stamp(s_k, from_node=from_node, to_node=to_node, I_value=I_eq)

    position = elements["num_nets"]
    for i, element in enumerate(elements["vsource_list"]):
        from_node = element["from"]
        to_node = element["to"]
        vdc_num = position + i + 1
        v_value = element["value"] * Convert_unit_to_value[element["unit"]]
        V.append("I_" + element["instance_name"])
        Element_stamps.vdc_stamp(j_f, s_k, from_node=from_node, to_node=to_node, v_value=v_value, vdc_num=vdc_num)


    #remove the zeroth line
    j_f = j_f[1:len(j_f), 1:len(j_f)]
    s_k = s_k[1:len(s_k)]
    return j_f, V, s_k

def matrix_formulation_AC(elements, freq):
    n = elements["num_nets"] \
        + elements["vsource_list"].__len__() \
        + elements["inductor_list"].__len__()\
        + elements["vcvs_list"].__len__() \
        + elements["cccs_list"].__len__() \
        + 2*elements["ccvs_list"].__len__() \
        + elements["opamp_list"].__len__()
    Y = np.zeros([n+1, n+1], dtype="complex")
    J = np.zeros([n+1, 1])
    V = []
    # Construct Unknown Vector 'V'
    for i in range(elements["num_nets"]):
        V.append(f"V{i+1}")

    for element in elements["resistor_list"]:
        from_node = element["from"]
        to_node = element["to"]
        value = element["value"] * Convert_unit_to_value[element["unit"]]
        Element_stamps.res_stamp(Y, from_node = from_node , to_node= to_node , res_value= value)

    for element in elements["isource_list"]:
        from_node = element["from"]
        to_node = element["to"]
        value = element["value"] * Convert_unit_to_value[element["unit"]]
        Element_stamps.idc_stamp(J, from_node = from_node , to_node= to_node , I_value= value)

    for element in elements["capacitor_list"]:
        # the cap is a gm = jwc
        from_node = element["from"]
        to_node = element["to"]
        if freq == 0:
            pass
        else:
            value = 1/(1j*2*np.pi*freq*element["value"] * Convert_unit_to_value[element["unit"]])
            Element_stamps.res_stamp(Y, from_node = from_node , to_node= to_node , res_value= value)

    position = elements["num_nets"]
    for i, element in enumerate(elements["vsource_list"]):
        from_node = element["from"]
        to_node = element["to"]
        vdc_num = position + i + 1
        v_value = element["value"] * Convert_unit_to_value[element["unit"]]
        V.append("I_" + element["instance_name"])
        Element_stamps.vdc_stamp(Y, J, from_node = from_node , to_node= to_node , v_value= v_value , vdc_num = vdc_num)

    position += elements["vsource_list"].__len__()
    for i, element in enumerate(elements["inductor_list"]):
        # the ind is a res = jwL
        from_node = element["from"]
        to_node = element["to"]
        ind_num = position + i + 1
        V.append("I_" + element["instance_name"])
        Element_stamps.vdc_stamp(Y, J, from_node=from_node, to_node=to_node, v_value=0, vdc_num=ind_num) 
        if freq == 0:
            value = np.inf
        else:
            value = 1/(1j*freq*2*np.pi*element["value"] * Convert_unit_to_value[element["unit"]])
        Element_stamps.res_stamp(Y, from_node = 0 , to_node= ind_num , res_value= -value)

    for i, element in enumerate(elements["vccs_list"]):
        from_node_1 = element["from_1"]
        to_node_1 = element["to_1"]
        from_node_2 = element["from_2"]
        to_node_2 = element["to_2"]
        gm = element["value"] * Convert_unit_to_value[element["unit"]]
        Element_stamps.vccs_stamp(Y, from_nodes = (from_node_1,from_node_2) , to_nodes= (to_node_1,to_node_2) , gm= gm)

    for i, element in enumerate(elements["vcvs_list"]):
        from_node_1 = element["from_1"]
        to_node_1 = element["to_1"]
        from_node_2 = element["from_2"]
        to_node_2 = element["to_2"]
        vcvs_num = position + i + 1
        A = element["value"] * Convert_unit_to_value[element["unit"]]

        V.append("I_" + element["instance_name"])
        Element_stamps.vcvs_stamp(Y, from_nodes = (from_node_1,from_node_2) , to_nodes= (to_node_1,to_node_2) , A=A, vcvs_num = vcvs_num)

    position += elements["vcvs_list"].__len__()
    for i, element in enumerate(elements["cccs_list"]):
        from_node_1 = element["from_1"]
        to_node_1 = element["to_1"]
        from_node_2 = element["from_2"]
        to_node_2 = element["to_2"]
        cccs_num = position + i + 1
        A = element["value"] * Convert_unit_to_value[element["unit"]]
        V.append("I_" + element["instance_name"])

        Element_stamps.cccs_stamp(Y, from_nodes = (from_node_1,from_node_2) , to_nodes= (to_node_1,to_node_2) , A=A, cccs_num = cccs_num)

    position += elements["cccs_list"].__len__()
    for i, element in enumerate(elements["ccvs_list"]):
        from_node_1 = element["from_1"]
        to_node_1 = element["to_1"]
        from_node_2 = element["from_2"]
        to_node_2 = element["to_2"]
        ccvs_num = position + i + 1
        Rm = element["value"] * Convert_unit_to_value[element["unit"]]

        V.append("I_" + element["instance_name"] + "_1")
        V.append("I_" + element["instance_name"] + "_2")
        Element_stamps.ccvs_stamp(Y, from_nodes = (from_node_1,from_node_2) , to_nodes= (to_node_1,to_node_2) , Rm=Rm, ccvs_num = ccvs_num)

    position += elements["ccvs_list"].__len__()
    for i, element in enumerate(elements["opamp_list"]):
        neg_terminal = element["neg_terminal"]
        pos_terminal = element["pos_terminal"]
        out_terminal = element["out_terminal"]
        opamp_num = position + i + 1

        V.append("I_" + element["instance_name"])
        Element_stamps.opamp_stamp(Y, neg_terminal = neg_terminal , pos_terminal = pos_terminal , out_terminal = out_terminal, opamp_num = opamp_num)
    # remove the zeroth line
    Y = Y[1:len(Y), 1:len(Y)]
    J = J[1:len(J)]
    return Y, V, J

def matrix_formulation_pre_tran(elements):
    n = elements["num_nets"] \
        + elements["vsource_list"].__len__()\
        + elements["vcos_list"].__len__()
    Y = np.zeros([n + 1, n + 1])
    J = np.zeros([n + 1, 1])
    V = []
    # Construct Unknown Vector 'V'
    for i in range(elements["num_nets"]):
        V.append(f"V{i + 1}")

    for element in elements["resistor_list"]:
        from_node = element["from"]
        to_node = element["to"]
        value = element["value"] * Convert_unit_to_value[element["unit"]]
        Element_stamps.res_stamp(Y, from_node=from_node, to_node=to_node, res_value=value)

    for element in elements["isource_list"]:
        from_node = element["from"]
        to_node = element["to"]
        if element["type"] == "step":
            I_value = 0
        else:
            I_value = element["value"] * Convert_unit_to_value[element["unit"]]
        Element_stamps.idc_stamp(J, from_node=from_node, to_node=to_node, I_value=I_value)

    for element in elements["capacitor_list"]:
        # o.c is an Isource with I = 0
        from_node = element["from"]
        to_node = element["to"]
        value = 0
        Element_stamps.idc_stamp(J, from_node=from_node, to_node=to_node, I_value=value)

    position = elements["num_nets"]
    for i, element in enumerate(elements["vsource_list"]):
        from_node = element["from"]
        to_node = element["to"]
        vdc_num = position + i + 1

        if element["type"] == "step":
            v_value = 0
        else:
            v_value = element["value"] * Convert_unit_to_value[element["unit"]]

        V.append("I_" + element["instance_name"])
        Element_stamps.vdc_stamp(Y, J, from_node=from_node, to_node=to_node, v_value=v_value, vdc_num=vdc_num)

    position +=  elements["vsource_list"].__len__()
    for i, element in enumerate(elements["vcos_list"]):
        from_node = element["from"]
        to_node = element["to"]
        vdc_num = position + i + 1
        freq = element["freq"] * Convert_unit_to_value[element["f_unit"]]
        v_value = 0
        V.append("I_" + element["instance_name"])
        Element_stamps.vdc_stamp(Y, J, from_node=from_node, to_node=to_node, v_value=v_value, vdc_num=vdc_num)
    # remove the zeroth line
    Y = Y[1:len(Y), 1:len(Y)]
    J = J[1:len(J)]
    return Y, V, J

def matrix_formulation_pre_tran_non_linear(elements, last_iter_results):
    n = elements["num_nets"] \
        + elements["vsource_list"].__len__()\
        + elements["vcos_list"].__len__()
    j_f = np.zeros([n + 1, n + 1])
    s_k = np.zeros([n + 1, 1])
    V = []

    # add the volt of the ground
    last_iter_results = np.vstack((np.array([0]), last_iter_results))
    # Construct Unknown Vector 'V'
    for i in range(elements["num_nets"]):
        V.append(f"V{i + 1}")       #add the voltages of the nets, the currents is added later in each element needs it

    for element in elements["resistor_list"]:
        from_node = element["from"]
        to_node = element["to"]
        value = element["value"] * Convert_unit_to_value[element["unit"]]
        Element_stamps.res_stamp(j_f, from_node=from_node, to_node=to_node, res_value=value)

    for element in elements["isource_list"]:
        from_node = element["from"]
        to_node = element["to"]

        if element["type"] == "step":
            I_value = 0
        else:
            I_value = element["value"] * Convert_unit_to_value[element["unit"]]
        Element_stamps.idc_stamp(s_k, from_node=from_node, to_node=to_node, I_value=I_value)

    for element in elements["capacitor_list"]:
        # o.c is an Isource with I = 0
        from_node = element["from"]
        to_node = element["to"]
        value = 0
        Element_stamps.idc_stamp(j_f, from_node=from_node, to_node=to_node, I_value=value)

    for i, element in enumerate(elements["diode_list"]):
        from_node = element["from"]
        to_node = element["to"]
        temp = 27
        I_s = 1e-13
        V_T = 25.8e-3

        volt_drop = float(last_iter_results[from_node] - last_iter_results[to_node])
        G_eq = (I_s/V_T)*np.exp(volt_drop/V_T)
        I_eq = I_s*(np.exp(volt_drop/V_T) - 1) - volt_drop*G_eq

        Element_stamps.res_stamp(j_f, from_node=from_node, to_node=to_node, res_value=1/G_eq)
        Element_stamps.idc_stamp(s_k, from_node=from_node, to_node=to_node, I_value=I_eq)

    position = elements["num_nets"]
    for i, element in enumerate(elements["vsource_list"]):
        from_node = element["from"]
        to_node = element["to"]
        vdc_num = position + i + 1

        if element["type"] == "step":
            v_value = 0
        else:
            v_value = element["value"] * Convert_unit_to_value[element["unit"]]

        V.append("I_" + element["instance_name"])
        Element_stamps.vdc_stamp(j_f, s_k, from_node=from_node, to_node=to_node, v_value=v_value, vdc_num=vdc_num)

    position +=  elements["vsource_list"].__len__()
    for i, element in enumerate(elements["vcos_list"]):
        from_node = element["from"]
        to_node = element["to"]
        vdc_num = position + i + 1
        freq = element["freq"] * Convert_unit_to_value[element["f_unit"]]
        v_value = 0
        V.append("I_" + element["instance_name"])
        Element_stamps.vdc_stamp(j_f, s_k, from_node=from_node, to_node=to_node, v_value=v_value, vdc_num=vdc_num)

    #remove the zeroth line
    j_f = j_f[1:len(j_f), 1:len(j_f)]
    s_k = s_k[1:len(s_k)]
    return j_f, V, s_k

def matrix_formulation_tran(elements, time_step, last_step_result, time):
    n = elements["num_nets"] \
        + elements["vsource_list"].__len__() \
        + elements["vcos_list"].__len__() \
        + elements["inductor_list"].__len__() \
        + elements["vcvs_list"].__len__() \
        + elements["cccs_list"].__len__() \
        + 2 * elements["ccvs_list"].__len__()
    Y = np.zeros([n + 1, n + 1])
    J = np.zeros([n + 1, 1])
    V = []
    # add the volt of the ground
    results = np.vstack((np.array([0]), last_step_result))

    # Construct Unknown Vector 'V'
    for i in range(elements["num_nets"]):
        V.append(f"V{i + 1}")

    for element in elements["resistor_list"]:
        from_node = element["from"]
        to_node = element["to"]
        value = element["value"] * Convert_unit_to_value[element["unit"]]
        Element_stamps.res_stamp(Y, from_node=from_node, to_node=to_node, res_value=value)

    for element in elements["isource_list"]:
        from_node = element["from"]
        to_node = element["to"]

        if element["type"] == "step":
            I_value = element["value"] * Convert_unit_to_value[element["unit"]]
        else:
            I_value = element["value"] * Convert_unit_to_value[element["unit"]]

        Element_stamps.idc_stamp(J, from_node=from_node, to_node=to_node, I_value=I_value)

    for element in elements["capacitor_list"]:
        from_node = element["from"]
        to_node = element["to"]
        cap_value = (element["value"] * Convert_unit_to_value[element["unit"]])
        res_value = time_step/ cap_value
        Element_stamps.res_stamp(Y, from_node=from_node, to_node=to_node, res_value=res_value)

        previous_volt_drop = float(results[from_node] - results[to_node])

        I_value = -1*cap_value * previous_volt_drop / time_step
        Element_stamps.idc_stamp(J, from_node=from_node, to_node=to_node, I_value=I_value)

    position = elements["num_nets"]
    for i, element in enumerate(elements["vsource_list"]):
        from_node = element["from"]
        to_node = element["to"]
        vdc_num = position + i + 1

        if element["type"] == "step":
            v_value = element["value"] * Convert_unit_to_value[element["unit"]]
        else:
            v_value = element["value"] * Convert_unit_to_value[element["unit"]]

        V.append("I_" + element["instance_name"])
        Element_stamps.vdc_stamp(Y, J, from_node=from_node, to_node=to_node, v_value=v_value, vdc_num=vdc_num)

    position += elements["vsource_list"].__len__()
    for i, element in enumerate(elements["vcos_list"]):
        from_node = element["from"]
        to_node = element["to"]
        vdc_num = position + i + 1
        freq = element["freq"] * Convert_unit_to_value[element["f_unit"]]
        if element["type"] == "cos":
            v_value = np.cos(time * 2 * np.pi * freq) * element["value"] * Convert_unit_to_value[element["unit"]]
        elif element["type"] == "sin":
            v_value = np.sin(time * 2 * np.pi * freq) * element["value"] * Convert_unit_to_value[element["unit"]]

        V.append("I_" + element["instance_name"])
        Element_stamps.vdc_stamp(Y, J, from_node=from_node, to_node=to_node, v_value=v_value, vdc_num=vdc_num)

    position += elements["vcos_list"].__len__()
    for i, element in enumerate(elements["inductor_list"]):
        pass

    for i, element in enumerate(elements["vccs_list"]):
        from_node_1 = element["from_1"]
        to_node_1 = element["to_1"]
        from_node_2 = element["from_2"]
        to_node_2 = element["to_2"]
        gm = element["value"] * Convert_unit_to_value[element["unit"]]
        Element_stamps.vccs_stamp(Y, from_nodes=(from_node_1, from_node_2), to_nodes=(to_node_1, to_node_2), gm=gm)

    for i, element in enumerate(elements["vcvs_list"]):
        from_node_1 = element["from_1"]
        to_node_1 = element["to_1"]
        from_node_2 = element["from_2"]
        to_node_2 = element["to_2"]
        vcvs_num = position + i + 1
        A = element["value"] * Convert_unit_to_value[element["unit"]]

        V.append("I_" + element["instance_name"])
        Element_stamps.vcvs_stamp(Y, from_nodes=(from_node_1, from_node_2), to_nodes=(to_node_1, to_node_2), A=A,
                                  vcvs_num=vcvs_num)

    position += elements["vcvs_list"].__len__()
    for i, element in enumerate(elements["cccs_list"]):
        from_node_1 = element["from_1"]
        to_node_1 = element["to_1"]
        from_node_2 = element["from_2"]
        to_node_2 = element["to_2"]
        cccs_num = position + i + 1
        A = element["value"] * Convert_unit_to_value[element["unit"]]
        V.append("I_" + element["instance_name"])

        Element_stamps.cccs_stamp(Y, from_nodes=(from_node_1, from_node_2), to_nodes=(to_node_1, to_node_2), A=A,
                                  cccs_num=cccs_num)

    position += elements["cccs_list"].__len__()
    for i, element in enumerate(elements["ccvs_list"]):
        from_node_1 = element["from_1"]
        to_node_1 = element["to_1"]
        from_node_2 = element["from_2"]
        to_node_2 = element["to_2"]
        ccvs_num = position + i + 1
        Rm = element["value"] * Convert_unit_to_value[element["unit"]]

        V.append("I_" + element["instance_name"] + "_1")
        V.append("I_" + element["instance_name"] + "_2")
        Element_stamps.ccvs_stamp(Y, from_nodes=(from_node_1, from_node_2), to_nodes=(to_node_1, to_node_2), Rm=Rm,
                                  ccvs_num=ccvs_num)
    # remove the zeroth line
    Y = Y[1:len(Y), 1:len(Y)]
    J = J[1:len(J)]
    return Y, V, J

def matrix_formulation_tran_non_linear(elements, time_step, last_step_result, last_iter_results, time):
    n = elements["num_nets"] \
        + elements["vsource_list"].__len__() \
        + elements["vcos_list"].__len__()
    j_f = np.zeros([n + 1, n + 1])
    s_k = np.zeros([n + 1, 1])
    V = []
    # add the volt of the ground
    last_step_result = np.vstack((np.array([0]), last_step_result))
    last_iter_results = np.vstack((np.array([0]), last_iter_results))
    # Construct Unknown Vector 'V'
    for i in range(elements["num_nets"]):
        V.append(f"V{i + 1}")

    for element in elements["resistor_list"]:
        from_node = element["from"]
        to_node = element["to"]
        value = element["value"] * Convert_unit_to_value[element["unit"]]
        Element_stamps.res_stamp(j_f, from_node=from_node, to_node=to_node, res_value=value)

    for element in elements["isource_list"]:
        from_node = element["from"]
        to_node = element["to"]

        if element["type"] == "step":
            I_value = element["value"] * Convert_unit_to_value[element["unit"]]
        else:
            I_value = element["value"] * Convert_unit_to_value[element["unit"]]

        Element_stamps.idc_stamp(s_k, from_node=from_node, to_node=to_node, I_value=I_value)

    for i, element in enumerate(elements["diode_list"]):
        from_node = element["from"]
        to_node = element["to"]
        temp = 27               # temperature
        I_s = 1e-13             # Is : reverse saturation current
        V_T = 25.8e-3           # VT : thermal voltage KT/q
        m = 0.3                 # m  : Grading coefficient
        V_j = 0.7               # Vj : built in potential (0.7 for silicon)
        CJ0 = 2e-9              # CJ0: junction capacitance @ equlibrium (vd = 0)
        FC = 0.5
        F2 = (1-FC)**(1+m)
        F3 = (1-FC)*(1+m)
        tau_d = 40e-9            # tau_d : transient time

        last_step_volt_drop = float(last_step_result[from_node] - last_step_result[to_node])
        last_iter_volt_drop = float(last_iter_results[from_node] - last_iter_results[to_node])

        #static model
        G_eq = (I_s/V_T)*np.exp(last_iter_volt_drop / V_T)
        I_eq = I_s * (np.exp(last_iter_volt_drop / V_T) - 1) - last_iter_volt_drop * G_eq

        Element_stamps.res_stamp(j_f, from_node=from_node, to_node=to_node, res_value=1/G_eq)
        Element_stamps.idc_stamp(s_k, from_node=from_node, to_node=to_node, I_value=I_eq)
        #dynamic model

        if last_iter_volt_drop < FC * V_j :
            cj = CJ0*(1-(last_iter_volt_drop/V_j))**-m
            dcj_dv = CJ0 * (m / V_j) * ((1 - (last_iter_volt_drop / V_j)) ** -m - 1)
        else:
            cj = (CJ0/F2)*(F3 + m*last_iter_volt_drop/V_j)
            dcj_dv = (CJ0/F2) * (m / V_j)

        cd = tau_d *  I_s * (np.exp(last_iter_volt_drop / V_T) - 1)            #capacitance due to the diffusion
        dcd_dv = tau_d *  G_eq

        ct = cj + cd
        dct_dv = dcj_dv + dcd_dv

        Gc_eq = (1/time_step) * dct_dv * (last_iter_volt_drop - last_step_volt_drop) + (1/time_step) * ct
        Ic_eq = (1/time_step) * ct * (last_iter_volt_drop - last_step_volt_drop) - Gc_eq*last_iter_volt_drop
        Element_stamps.res_stamp(j_f, from_node=from_node, to_node=to_node, res_value=1/Gc_eq)
        Element_stamps.idc_stamp(s_k, from_node=from_node, to_node=to_node, I_value=Ic_eq)


    for element in elements["capacitor_list"]:
        from_node = element["from"]
        to_node = element["to"]
        cap_value = (element["value"] * Convert_unit_to_value[element["unit"]])
        res_value = time_step/ cap_value
        #the finite difference method used is BE and norton model
        Element_stamps.res_stamp(j_f, from_node=from_node, to_node=to_node, res_value=res_value)

        previous_volt_drop = float(last_step_result[from_node] - last_step_result[to_node])
        I_value = -1*cap_value * previous_volt_drop / time_step

        Element_stamps.idc_stamp(s_k, from_node=from_node, to_node=to_node, I_value=I_value)

    position = elements["num_nets"]
    for i, element in enumerate(elements["vsource_list"]):
        from_node = element["from"]
        to_node = element["to"]
        vdc_num = position + i + 1

        if element["type"] == "step":
            v_value = element["value"] * Convert_unit_to_value[element["unit"]]
        else:
            v_value = element["value"] * Convert_unit_to_value[element["unit"]]

        V.append("I_" + element["instance_name"])
        Element_stamps.vdc_stamp(j_f, s_k, from_node=from_node, to_node=to_node, v_value=v_value, vdc_num=vdc_num)

    position +=  elements["vsource_list"].__len__()
    for i, element in enumerate(elements["vcos_list"]):
        from_node = element["from"]
        to_node = element["to"]
        vdc_num = position + i + 1
        freq = element["freq"] * Convert_unit_to_value[element["f_unit"]]
        if element["type"] == "cos":
            v_value = np.cos(time * 2 * np.pi * freq) * element["value"] * Convert_unit_to_value[element["unit"]]
        elif element["type"] == "sin":
            v_value = np.sin(time * 2 * np.pi * freq) * element["value"] * Convert_unit_to_value[element["unit"]]

        V.append("I_" + element["instance_name"])
        Element_stamps.vdc_stamp(j_f, s_k, from_node=from_node, to_node=to_node, v_value=v_value, vdc_num=vdc_num)

    # remove the zeroth line
    j_f = j_f[1:len(j_f), 1:len(j_f)]
    s_k = s_k[1:len(s_k)]
    return j_f, V, s_k

def Divide_Result_Matrix(Solution_Matrix: np.array, V: List) -> Dict:
    Result_Dict = {}
    for i in range(len(V)):
        Result_Dict[V[i]] = Solution_Matrix[i, :]
    return Result_Dict

def Plot_Output(Plot_name, frequencies, Result):
    plt.figure()
    for i, val in enumerate(Plot_name):
        # plt.plot(frequencies, 20*np.log10(Result[val]), 'r')
        plt.plot(frequencies, Result[val])
        # plt.xlabel('Frequency (Hz)')
        plt.xlabel('Time (sec)')
        plt.ylabel('Amplitude')
        plt.title(f"{val} Curve")
        # plt.xscale('log')
        plt.grid(True)
    plt.show()