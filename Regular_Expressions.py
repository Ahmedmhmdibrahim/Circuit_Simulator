from typing import List, Dict
import re


def Resistor_Regx(resistor_line: str) -> bool:
    """
    This Function determine if resistor line match standard of netlist structure or not
    """
    pattern = r"^[A-Z|a-z|_][^\s]*\s+(R|r)esistor\s+\d+\s+\d+\s+\d+(\s*|(T|G|M|K|m|u|n|p|f))\s*$"
    return False if re.match(pattern, resistor_line) == None else True
def Diode_Regx(diode_line: str) -> bool:
    """
    This Function determine if resistor line match standard of netlist structure or not
    """
    pattern = r"^[A-Z|a-z|_][^\s]*\s+(D|d)iode\s+\d+\s+\d+\s*$" # todo: check if it is right
    return False if re.match(pattern, diode_line) == None else True

def Capacitor_Regx(capacitor_line: str) -> bool:
    """
    This Function determine if capacitor line match standard of netlist structure or not
    """
    pattern = r"^[A-Z|a-z|_][^\s]*\s+(C|c)apacitor\s+\d+\s+\d+\s+\d+(\s*|(T|G|M|K|m|u|n|p|f))\s*$"
    return False if re.match(pattern, capacitor_line) == None else True

def Inductor_Regx(voltage_dc_line: str) -> bool:
    """
    This Function determine if inductor source line match standard of netlist structure or not
    """
    pattern = r"^[A-Z|a-z|_][^\s]*\s+(I|i)nductor\s+\d+\s+\d+\s+\d+(\s*|(T|G|M|K|m|u|n|p|f))\s*$"
    return False if re.match(pattern, voltage_dc_line) == None else True

def Voltage_DC_Regx(voltage_dc_line: str) -> bool:
    """
    This Function determine if dc-voltage source line match standard of netlist structure or not
    """
    pattern = r"^[A-Z|a-z|_][^\s]*\s+(V|v)source\s+\d+\s+\d+\s+(dc|ac|step)\s+\d+(\s*|(T|G|M|K|m|u|n|p|f))\s*$"
    return False if re.match(pattern, voltage_dc_line) == None else True
def Voltage_cos_Regx(voltage_dc_line: str) -> bool:
    """
    This Function determine if dc-voltage source line match standard of netlist structure or not
    """
    pattern = r"^[A-Z|a-z|_][^\s]*\s+(V|v)cos\s+\d+\s+\d+\s+(cos|sin)\s+\d+(\s*|m|u)\s+\d+(\s*|(T|G|M|K|m|u|n|p|f))\s*$"
    return False if re.match(pattern, voltage_dc_line) == None else True
def Current_DC_Regx(voltage_dc_line: str) -> bool:
    """
    This Function determine if dc-current source line match standard of netlist structure or not
    """
    pattern = r"^[A-Z|a-z|_][^\s]*\s+(I|i)source\s+\d+\s+\d+\s+(dc|ac|cos|step)\s+\d+(\s*|(T|G|M|K|m|u|n|p|f))\s*$"
    return False if re.match(pattern, voltage_dc_line) == None else True

def VCCS_Regx(vccs_line: str) -> bool:
    """
    This Function determine if vccs_line match standard of netlist structure or not
    """
    pattern = r"^[A-Z|a-z|_][^\s]*\s+(V|v)ccs\s+\d+\s+\d+\s+\d+\s+\d+\s+(dc|ac)\s+\d+(\s*|(T|G|M|K|m|u|n|p|f))\s*$"
    return False if re.match(pattern, vccs_line) == None else True

def VCVS_Regx(vcvs_line: str) -> bool:
    """
    This Function determine if vcvs_line match standard of netlist structure or not
    """
    pattern = r"^[A-Z|a-z|_][^\s]*\s+(V|v)cvs\s+\d+\s+\d+\s+\d+\s+\d+\s+(dc|ac)\s+\d+(\s*|(T|G|M|K|m|u|n|p|f))\s*$"
    return False if re.match(pattern, vcvs_line) == None else True

def CCCS_Regx(cccs_line: str) -> bool:
    """
    This Function determine if cccs_line match standard of netlist structure or not
    """
    pattern = r"^[A-Z|a-z|_][^\s]*\s+(C|c)ccs\s+\d+\s+\d+\s+\d+\s+\d+\s+(dc|ac)\s+\d+(\s*|(T|G|M|K|m|u|n|p|f))\s*$"
    return False if re.match(pattern, cccs_line) == None else True

def CCVS_Regx(ccvs_line: str) -> bool:
    """
    This Function determine if ccvs_line match standard of netlist structure or not
    """
    pattern = r"^[A-Z|a-z|_][^\s]*\s+(C|c)cvs\s+\d+\s+\d+\s+\d+\s+\d+\s+(dc|ac)\s+\d+(\s*|(T|G|M|K|m|u|n|p|f))\s*$"
    return False if re.match(pattern, ccvs_line) == None else True

def OPAMP_Regx(opamp_line: str) -> bool:
    """
    This Function determine if opamp_line match standard of netlist structure or not
    """
    pattern = r"^[A-Z|a-z|_][^\s]*\s+(O|o)pamp\s+\d+\s+\d+\s+\d+\s*$"
    return False if re.match(pattern, opamp_line) == None else True

def DC_Analysis_Regx(dc_analysis_line: str) -> bool:
    """
    This Function determine if dc_analysis_line match standard of netlist structure or not
    """
    pattern = r"^[A-Z|a-z|_][^\s]*\s+(DC|dc)\s*$"
    return False if re.match(pattern, dc_analysis_line) == None else True

def AC_Analysis_Regx(ac_analysis_line: str) -> bool:
    """
    This Function determine if ac_analysis_line match standard of netlist structure or not
    """
    pattern = r"^[A-Z|a-z|_][^\s]*\s+(AC|ac)\s+\d+(\s*|(T|G|M|K|m|u|n|p|f))\s+\d+(\s*|(T|G|M|K|m|u|n|p|f))\s+\d+\s*$"
    return False if re.match(pattern, ac_analysis_line) == None else True

def Tran_Analysis_Regx(tran_analysis_line: str) -> bool:
    """
    This Function determine if tran_analysis_line match standard of netlist structure or not
    """
    pattern = r"^[A-Z|a-z|_][^\s]*\s+(TRAN|tran)\s+\d+(\s*|(T|G|M|K|m|u|n|p|f))\s+\d+(\s*|(T|G|M|K|m|u|n|p|f))\s*$"
    return False if re.match(pattern, tran_analysis_line) == None else True

def Plot_Output_Regx(plot_output_line: str) -> bool:
    """
    This Function determine if plot_output_line match standard of netlist structure or not
    """
    pattern = r"^(P|p)lot\s+[A-Z|a-z|_][^\s]*"
    #pattern = r"^plot\s+([A-Z|a-z|_][^\s]*\s){1,10}"
    return False if re.match(pattern, plot_output_line) == None else True