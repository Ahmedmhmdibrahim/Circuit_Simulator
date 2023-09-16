from typing import List, Dict
from Regular_Expressions import *
import pprint

def Resistor_Parsing(resistor_line: str) -> Dict:
    """
    This Function parse resistor_line to its parameters into dictionary as follow
    dict={"instance_name", "component_type", "from", "to", "type", "value", "unit"}
    """
    resistor_line_split = resistor_line.split()
    dict = {
        "instance_name": resistor_line_split[0],
        "component_type": resistor_line_split[1],
        "from": int(resistor_line_split[2]),
        "to": int(resistor_line_split[3]),
        "value": int(resistor_line_split[4][:-1]) if not resistor_line_split[4][-1].isdigit() else int(resistor_line_split[4]),
        "unit": resistor_line_split[4][-1] if not resistor_line_split[4][-1].isdigit() else "nothing"
    }
    return dict
def Diode_Parsing(diode_line: str) -> Dict:
    """
    This Function parse resistor_line to its parameters into dictionary as follow
    dict={"instance_name", "component_type", "from", "to", "type", "value", "unit"}
    """
    diode_line_split = diode_line.split()
    dict = {
        "instance_name": diode_line_split[0],
        "component_type": diode_line_split[1],
        "from": int(diode_line_split[2]),
        "to": int(diode_line_split[3]),
    }
    return dict

def Capacitor_Parsing(capacitor_line: str) -> Dict:
    """
    This Function parse capacitor_line to its parameters into dictionary as follow
    dict={"instance_name", "component_type", "from", "to", "type", "value", "unit"}
    """
    capacitor_line_split = capacitor_line.split()
    dict = {
        "instance_name": capacitor_line_split[0],
        "component_type": capacitor_line_split[1],
        "from": int(capacitor_line_split[2]),
        "to": int(capacitor_line_split[3]),
        "value": int(capacitor_line_split[4][:-1]) if not capacitor_line_split[4][-1].isdigit() else int(capacitor_line_split[4]),
        "unit": capacitor_line_split[4][-1] if not capacitor_line_split[4][-1].isdigit() else "nothing"
    }
    return dict

def Inductor_Parsing(inductor_line: str) -> Dict:
    """
    This Function parse inductor_line to its parameters into dictionary as follow
    dict={"instance_name", "component_type", "from", "to", "type", "value", "unit"}
    """
    inductor_line_split = inductor_line.split()
    dict = {
        "instance_name": inductor_line_split[0],
        "component_type": inductor_line_split[1],
        "from": int(inductor_line_split[2]),
        "to": int(inductor_line_split[3]),
        "value": int(inductor_line_split[4][:-1]) if not inductor_line_split[4][-1].isdigit() else int(inductor_line_split[4]),
        "unit": inductor_line_split[4][-1] if not inductor_line_split[4][-1].isdigit() else "nothing"
    }
    return dict

def Voltage_dc_Parsing(voltage_dc_line: str) -> Dict:
    """
    This Function parse voltage_dc_line to its parameters into dictionary as follows
    dict={"instance_name", "component_type", "from", "to", "type", "value", "unit"}
    """
    voltage_dc_line_split = voltage_dc_line.split()
    dict = {
        "instance_name": voltage_dc_line_split[0],
        "component_type": voltage_dc_line_split[1],
        "from": int(voltage_dc_line_split[2]),
        "to": int(voltage_dc_line_split[3]),
        "type": voltage_dc_line_split[4],
        # TODO : the type must be lower case or change from stamp
        "value": int(voltage_dc_line_split[5][:-1]) if not voltage_dc_line_split[5][-1].isdigit() else int(voltage_dc_line_split[5]),
        "unit": voltage_dc_line_split[5][-1] if not voltage_dc_line_split[5][-1].isdigit() else "nothing"
    }
    return dict
def Voltage_cos_Parsing(voltage_dc_line: str) -> Dict:
    """
    This Function parse voltage_dc_line to its parameters into dictionary as follows
    dict={"instance_name", "component_type", "from", "to", "type", "value", "unit" , "freq", "f_unit"}
    """
    voltage_dc_line_split = voltage_dc_line.split()
    dict = {
        "instance_name": voltage_dc_line_split[0],
        "component_type": voltage_dc_line_split[1],
        "from": int(voltage_dc_line_split[2]),
        "to": int(voltage_dc_line_split[3]),
        "type": voltage_dc_line_split[4],
        "value": int(voltage_dc_line_split[5][:-1]) if not voltage_dc_line_split[5][-1].isdigit() else int(voltage_dc_line_split[5]),
        "unit": voltage_dc_line_split[5][-1] if not voltage_dc_line_split[5][-1].isdigit() else "nothing",
        "freq": int(voltage_dc_line_split[6][:-1]) if not voltage_dc_line_split[6][-1].isdigit() else int(voltage_dc_line_split[6]),
        "f_unit":voltage_dc_line_split[6][-1] if not voltage_dc_line_split[6][-1].isdigit() else "nothing"
    }
    return dict
def Current_dc_Parsing(current_dc_line: str) -> Dict:
    """
    This Function parse current_dc_line to its parameters into dictionary as follow
    dict={"instance_name", "component_type", "from", "to", "type", "value", "unit"}
    """
    current_dc_line_split = current_dc_line.split()
    dict = {
        "instance_name": current_dc_line_split[0],
        "component_type": current_dc_line_split[1],
        "from": int(current_dc_line_split[2]),
        "to": int(current_dc_line_split[3]),
        # TODO : the type must be lower case or change from stamp
        "type": current_dc_line_split[4],
        "value": int(current_dc_line_split[5][:-1]) if not current_dc_line_split[5][-1].isdigit() else int(current_dc_line_split[5]),
        "unit": current_dc_line_split[5][-1] if not current_dc_line_split[5][-1].isdigit() else "nothing"
    }
    return dict

def VCCS_Parsing(vccs_line: str) -> Dict:
    """
    This Function parse vccs_line to its parameters into dictionary as follow
    dict={"instance_name", "component_type", "from_1", "to_1", "from_2", "to_2", "type", "value", "unit"}
    """
    vccs_line_split = vccs_line.split()
    dict = {
        "instance_name": vccs_line_split[0],
        "component_type": vccs_line_split[1],
        "from_1": int(vccs_line_split[2]),
        "to_1": int(vccs_line_split[3]),
        "from_2": int(vccs_line_split[4]),
        "to_2": int(vccs_line_split[5]),
        # TODO : the type must be lower case or change from stamp
        "type": vccs_line_split[6],
        "value": int(vccs_line_split[7][:-1]) if not vccs_line_split[7][-1].isdigit() else int(vccs_line_split[7]),
        "unit": vccs_line_split[7][-1] if not vccs_line_split[7][-1].isdigit() else "nothing"
    }
    return dict

def VCVS_Parsing(vcvs_line: str) -> Dict:
    """
    This Function parse vcvs_line to its parameters into dictionary as follow
    dict={"instance_name", "component_type", "from_1", "to_1", "from_2", "to_2", "type", "value", "unit"}
    """
    vcvs_line_split = vcvs_line.split()
    dict = {
        "instance_name": vcvs_line_split[0],
        "component_type": vcvs_line_split[1],
        "from_1": int(vcvs_line_split[2]),
        "to_1": int(vcvs_line_split[3]),
        "from_2": int(vcvs_line_split[4]),
        "to_2": int(vcvs_line_split[5]),
        "type": vcvs_line_split[6],
        "value": int(vcvs_line_split[7][:-1]) if not vcvs_line_split[7][-1].isdigit() else int(vcvs_line_split[7]),
        "unit": vcvs_line_split[7][-1] if not vcvs_line_split[7][-1].isdigit() else "nothing"
    }
    return dict

def CCCS_Parsing(cccs_line: str) -> Dict:
    """
    This Function parse cccs_line to its parameters into dictionary as follow
    dict={"instance_name", "component_type", "from_1", "to_1", "from_2", "to_2", "type", "value", "unit"}
    """
    cccs_line_split = cccs_line.split()
    dict = {
        "instance_name": cccs_line_split[0],
        "component_type": cccs_line_split[1],
        "from_1": int(cccs_line_split[2]),
        "to_1": int(cccs_line_split[3]),
        "from_2": int(cccs_line_split[4]),
        "to_2": int(cccs_line_split[5]),
        "type": cccs_line_split[6],
        "value": int(cccs_line_split[7][:-1]) if not cccs_line_split[7][-1].isdigit() else int(cccs_line_split[7]),
        "unit": cccs_line_split[7][-1] if not cccs_line_split[7][-1].isdigit() else "nothing"
    }
    return dict

def CCVS_Parsing(ccvs_line: str) -> Dict:
    """
    This Function parse ccvs_line to its parameters into dictionary as follow
    dict={"instance_name", "component_type", "from_1", "to_1", "from_2", "to_2", "type", "value", "unit"}
    """
    ccvs_line_split = ccvs_line.split()
    dict = {
        "instance_name": ccvs_line_split[0],
        "component_type": ccvs_line_split[1],
        "from_1": int(ccvs_line_split[2]),
        "to_1": int(ccvs_line_split[3]),
        "from_2": int(ccvs_line_split[4]),
        "to_2": int(ccvs_line_split[5]),
        "type": ccvs_line_split[6],
        "value": int(ccvs_line_split[7][:-1]) if not ccvs_line_split[7][-1].isdigit() else int(ccvs_line_split[7]),
        "unit": ccvs_line_split[7][-1] if not ccvs_line_split[7][-1].isdigit() else "nothing"
    }
    return dict

def OPAMP_Parsing(opamp_line: str) -> Dict:
    """
    This Function parse opamp_line to its parameters into dictionary as follow
    dict={"instance_name", "component_type", "neg_terminal", "pos_terminal", "out_terminal"}
    """
    opamp_line_split = opamp_line.split()
    dict = {
        "instance_name": opamp_line_split[0],
        "component_type": opamp_line_split[1],
        "neg_terminal": int(opamp_line_split[2]),
        "pos_terminal": int(opamp_line_split[3]),
        "out_terminal": int(opamp_line_split[4]),
    }
    return dict

def DC_Analysis_Parsing(dc_analysis_line: str) -> Dict:
    """
    This Function parse dc_analysis_line to its parameters into dictionary as follow
    dict={"analysis_name", "analysis_type"}
    """
    dc_analysis_line_split = dc_analysis_line.split()
    dict = {
        "analysis_name": dc_analysis_line_split[0],
        "analysis_type": dc_analysis_line_split[1]
    }
    return dict

def AC_Analysis_Parsing(ac_analysis_line: str) -> Dict:
    """
    This Function parse ac_analysis_line to its parameters into dictionary as follow
    dict={"analysis_name", "analysis_type", "freq_start", "freq_start_unit", "freq_stop", "freq_stop_unit", "points_per_dec"}
    """
    ac_analysis_line_split = ac_analysis_line.split()
    dict = {
        "analysis_name": ac_analysis_line_split[0],
        "analysis_type": ac_analysis_line_split[1],
        "freq_start": int(ac_analysis_line_split[2][:-1]) if not ac_analysis_line_split[2][-1].isdigit() else int(ac_analysis_line_split[2]),
        "freq_start_unit": ac_analysis_line_split[2][-1] if not ac_analysis_line_split[2][-1].isdigit() else "nothing",
        "freq_stop": int(ac_analysis_line_split[3][:-1]) if not ac_analysis_line_split[3][-1].isdigit() else int(ac_analysis_line_split[3]),
        "freq_stop_unit": ac_analysis_line_split[3][-1] if not ac_analysis_line_split[3][-1].isdigit() else "nothing",
        "points_per_dec": int(ac_analysis_line_split[4])
    }
    return dict

def Tran_Analysis_Parsing(tran_analysis_line: str) -> Dict:
    """
    This Function parse tran_analysis_line to its parameters into dictionary as follows
    dict={"analysis_name", "analysis_type", "time_step", "time_step_unit, "stop_time", "stop_time_unit"}
    """
    tran_analysis_line_split = tran_analysis_line.split()
    dict = {
        "analysis_name": tran_analysis_line_split[0],
        "analysis_type": tran_analysis_line_split[1],
        "time_step": int(tran_analysis_line_split[2][:-1]) if not tran_analysis_line_split[2][-1].isdigit() else int(tran_analysis_line_split[2]),
        "time_step_unit": tran_analysis_line_split[2][-1] if not tran_analysis_line_split[2][-1].isdigit() else "nothing",
        "stop_time": int(tran_analysis_line_split[3][:-1]) if not tran_analysis_line_split[3][-1].isdigit() else int(tran_analysis_line_split[3]),
        "stop_time_unit": tran_analysis_line_split[3][-1] if not tran_analysis_line_split[3][-1].isdigit() else "nothing"
    }
    return dict

def Plot_Output_Parsing(plot_output_line: str) -> str:
    """
    This Function parse plot_output_line to its parameters into dictionary as follow
    dict={"plot_name"}
    """
    plot_output_line_split = plot_output_line.split()
    for i, val in enumerate(plot_output_line_split):
        if i != 0:
            return val

def Get_Number_of_Nets(circuit_dict: dict): # -> [int ,Dict] :
    """
    This Function determine and return the number of nets in the circuit
    """
    number_of_nets = 0
    for i, val in enumerate(circuit_dict):
        if circuit_dict[val] != [] and val != "num_nets" and val != "analysis" and val != 'plot_name':
            for j, v in enumerate(circuit_dict[val]):
                different_component = ['vccs', 'vcvs', 'cccs', 'ccvs', 'opamp']
                if v['component_type'] not in different_component:
                    number_of_nets = max(number_of_nets, v['from'], v['to'])
                elif v['component_type'] == 'opamp':
                    number_of_nets = max(number_of_nets, v['neg_terminal'], v['pos_terminal'], v['out_terminal'])
                else:
                    number_of_nets = max(number_of_nets, v['from_1'], v['to_1'], v['from_2'], v['to_2'])
        else:
            pass
    return number_of_nets

def parser(content: str) -> Dict:
    content_without_dashed_lines = []
    circuit_dict = {
        "analysis": [],
        "num_nets": int,
        "resistor_list": [],
        "diode_list": [],
        "vsource_list": [],
        "vcos_list": [],
        "isource_list": [],
        "capacitor_list": [],
        "inductor_list": [],
        "vccs_list": [],
        "vcvs_list": [],
        "cccs_list": [],
        "opamp_list":[],
        "ccvs_list": [],
        "plot_name": []
    }
    for i, val in enumerate(content):
        if '//' not in val:
            val = val+' ' if i != len(content)-1 else val
            content_without_dashed_lines.append(val)

    for i, val in enumerate(content_without_dashed_lines):
        if Resistor_Regx(val):
            circuit_dict["resistor_list"].append(Resistor_Parsing(val))
        elif Diode_Regx(val):
            circuit_dict["diode_list"].append(Diode_Parsing(val))
        elif Capacitor_Regx(val):
            circuit_dict["capacitor_list"].append(Capacitor_Parsing(val))
        elif Inductor_Regx(val):
            circuit_dict["inductor_list"].append(Inductor_Parsing(val))
        elif Voltage_DC_Regx(val):
            circuit_dict["vsource_list"].append(Voltage_dc_Parsing(val))
        elif Voltage_cos_Regx(val):
            circuit_dict["vcos_list"].append(Voltage_cos_Parsing(val))
        elif Current_DC_Regx(val):
            circuit_dict["isource_list"].append(Current_dc_Parsing(val))
        elif VCCS_Regx(val):
            circuit_dict["vccs_list"].append(VCCS_Parsing(val))
        elif VCVS_Regx(val):
            circuit_dict["vcvs_list"].append(VCVS_Parsing(val))
        elif CCCS_Regx(val):
            circuit_dict["cccs_list"].append(CCCS_Parsing(val))
        elif CCVS_Regx(val):
            circuit_dict["ccvs_list"].append(CCVS_Parsing(val))
        elif OPAMP_Regx(val):
            circuit_dict["opamp_list"].append(OPAMP_Parsing(val))
        elif DC_Analysis_Regx(val):
            circuit_dict["analysis"].append(DC_Analysis_Parsing(val))
        elif AC_Analysis_Regx(val):
            circuit_dict["analysis"].append(AC_Analysis_Parsing(val))
        elif Tran_Analysis_Regx(val):
            circuit_dict["analysis"].append(Tran_Analysis_Parsing(val))
        elif Plot_Output_Regx(val):
            circuit_dict["plot_name"].append(Plot_Output_Parsing(val))
        else:
            pass
            # TODO: Do something notify for error
    circuit_dict["num_nets"] = Get_Number_of_Nets(circuit_dict)
    return circuit_dict
