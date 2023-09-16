import pprint
import numpy as np
from File_fun import read_file
from Parsing_fun import parser
from Simulations import *
from Element_stamps import Convert_unit_to_value
from Solution import *

########################### Netlist Parsing ####################### 
Circuit_Matrix = parser(read_file('Netlist_3.txt', list))
# pprint.pprint(Circuit_Matrix)

# TODO : try and catch must be implemented here (if netlist not correct)
if Circuit_Matrix["analysis"][0]["analysis_type"] == "dc":
########################### DC Analysis ###########################
#     Y, V, J = matrix_formulation_OP(Circuit_Matrix)
#     Result = Solve_real_Linear_Matrix(Y, J)
#     for i in range(len(V)):
#         print(f"{V[i]} = {Result[i]}")
# elif Circuit_Matrix["analysis"][0]["analysis_type"] == "dc":      #just dummy code for linear DC

########################### nonlinear DC Analysis ###########################
    n_nets = Circuit_Matrix["num_nets"]                                             # actual number of nets -> actual KCL -> actual currents
    n = Circuit_Matrix["num_nets"] + Circuit_Matrix["vsource_list"].__len__()       # the size of the total MNA
    relative_tolerance = 0.001
    volt_absolute_tolerance = 10e-6
    current_absolute_tolerance = 10e-9

    x_k = np.zeros([n, 1])
    while True:
        # iterate to solve Gx+Hg(x)=s
        #                   j_f . x_k_new = s_k  , s_k = f(x_k)
        j_f, V, s_k = matrix_formulation_OP_non_linear(Circuit_Matrix, last_iter_results= x_k)

        x_k_new = Solve_real_Linear_Matrix(j_f, s_k)

        #the residue criterion
        f_x_k = j_f @ x_k - s_k
        I_max = max(s_k[:n_nets,0])
        mag_f_x_k = np.sqrt(np.sum(f_x_k ** 2))
        condition_R_C = ( mag_f_x_k < relative_tolerance * abs(I_max) + current_absolute_tolerance)
        #the update criterion
        error_vector = x_k_new - x_k
        error_mag = np.sqrt(np.sum(error_vector ** 2))
        max_nominal_solution = max(max(x_k) , max(x_k_new))
        condition_U_C = ( error_mag < relative_tolerance * abs(max_nominal_solution) + current_absolute_tolerance)

        maximum_delta = 0.05        #2VT = 0.0258 * 2 ~= 0.05

        #todo : make the damping criterion

        # x_k = x_k + maximum_delta * (error_vector > maximum_delta)
        # x_k = x_k - maximum_delta * (error_vector < -maximum_delta)
        # x_k += error_vector * (~(error_vector > maximum_delta) &  ~(error_vector < -maximum_delta))
        x_k = x_k_new

        if (condition_R_C and condition_U_C):
            break

    for i in range(len(V)):
            print(f"{V[i]} = {x_k[i]}")

elif Circuit_Matrix["analysis"][0]["analysis_type"] == "ac":
########################### AC Analysis ########################### 
    n = Circuit_Matrix["num_nets"] + Circuit_Matrix["vsource_list"].__len__() + Circuit_Matrix["inductor_list"].__len__() + Circuit_Matrix["opamp_list"].__len__()
    from_frequency = Circuit_Matrix["analysis"][0]["freq_start"] * Convert_unit_to_value[Circuit_Matrix["analysis"][0]["freq_start_unit"]]
    to_frequency = Circuit_Matrix["analysis"][0]["freq_stop"] * Convert_unit_to_value[Circuit_Matrix["analysis"][0]["freq_stop_unit"]]
    if from_frequency == 0:
        number_of_decades = int(np.log10(to_frequency/1)) + 1
    else:
        number_of_decades = int(np.log10(to_frequency / from_frequency))
    number_of_frequencies = Circuit_Matrix["analysis"][0]["points_per_dec"]*number_of_decades
    solution_vector = np.zeros([n, number_of_frequencies])
    frequencies = np.linspace(start=from_frequency, stop=to_frequency, num=number_of_frequencies)
    V = []
    for i, frq in enumerate(frequencies):
        Y, V, J = matrix_formulation_AC(Circuit_Matrix, frq)
        solution_vector[:, i, np.newaxis] = Solve_complex_Linear_Matrix(Y, J)
    Result = Divide_Result_Matrix(solution_vector, V)
    Plot_Output(Circuit_Matrix['plot_name'], frequencies, Result)
# TODO: ADD Opamp to transient 
elif Circuit_Matrix["analysis"][0]["analysis_type"] == "tran_":
########################### tran Analysis ###########################

    stop_time = Circuit_Matrix["analysis"][0]["stop_time"]*Convert_unit_to_value[Circuit_Matrix["analysis"][0]["stop_time_unit"]]
    time_step = Circuit_Matrix["analysis"][0]["time_step"]*Convert_unit_to_value[Circuit_Matrix["analysis"][0]["time_step_unit"]]
    steps = np.linspace(start=time_step, stop=stop_time, num=int((stop_time / time_step)))

    ## first step is an op
    n = Circuit_Matrix["num_nets"] + Circuit_Matrix["vsource_list"].__len__()
    Y, V, J = matrix_formulation_pre_tran(Circuit_Matrix)
    Result = Solve_real_Linear_Matrix(Y, J)
    solution_vector = np.zeros([n, len(steps) + 1])
    #V = []
    solution_vector[:, 0, np.newaxis] = Result

    #solution_vector[1,0] =0 # for step effect

    for i,step in enumerate(steps):
        Y ,V ,J = matrix_formulation_tran(Circuit_Matrix,
                                          time_step,
                                          last_step_result=solution_vector[:, i, np.newaxis],
                                          time = step)
        solution_vector[:, i + 1 , np.newaxis] = Solve_real_Linear_Matrix(Y, J)

    Result = Divide_Result_Matrix(solution_vector, V)
    # pprint.pprint(solution_vector)
    steps = np.hstack((np.array([0]) , steps))
    Plot_Output(Circuit_Matrix['plot_name'], steps, Result)

elif Circuit_Matrix["analysis"][0]["analysis_type"] == "tran":
########################### non-linear tran Analysis ###########################

    stop_time = Circuit_Matrix["analysis"][0]["stop_time"]*Convert_unit_to_value[Circuit_Matrix["analysis"][0]["stop_time_unit"]]
    time_step = Circuit_Matrix["analysis"][0]["time_step"]*Convert_unit_to_value[Circuit_Matrix["analysis"][0]["time_step_unit"]]
    steps = np.linspace(start=time_step, stop=stop_time, num=int((stop_time / time_step)))

    ## first step is an op
    n_nets = Circuit_Matrix["num_nets"]  # actual number of nets -> actual KCL -> actual currents
    n = Circuit_Matrix["num_nets"] \
        + Circuit_Matrix["vsource_list"].__len__()\
        + Circuit_Matrix["vcos_list"].__len__()# the size of the total MNA
    relative_tolerance = 0.001
    volt_absolute_tolerance = 10e-6
    current_absolute_tolerance = 10e-9

    x_k = np.zeros([n, 1])      #initial guess: zeros

    while True:
        # iterate to solve Gx+Hg(x)=s
        #                   j_f . x_k_new = s_k  , s_k = f(x_k)
        j_f, V, s_k = matrix_formulation_pre_tran_non_linear(Circuit_Matrix, last_iter_results=x_k)

        x_k_new = Solve_real_Linear_Matrix(j_f, s_k)

        # the residue criterion
        f_x_k = j_f @ x_k - s_k
        I_max = max(s_k[:n_nets, 0])
        mag_f_x_k = np.sqrt(np.sum(f_x_k ** 2))
        condition_R_C = (mag_f_x_k < relative_tolerance * abs(I_max) + current_absolute_tolerance)
        # the update criterion
        error_vector = x_k_new - x_k
        error_mag = np.sqrt(np.sum(error_vector ** 2))
        max_nominal_solution = max(max(x_k), max(x_k_new))
        condition_U_C = (error_mag < relative_tolerance * abs(max_nominal_solution) + current_absolute_tolerance)
        # maximum_delta = 0.05  # 2VT = 0.0258 * 2 ~= 0.05
        # todo : make the damping criterion
        # x_k = x_k + maximum_delta * (error_vector > maximum_delta)
        # x_k = x_k - maximum_delta * (error_vector < -maximum_delta)
        # x_k += error_vector * (~(error_vector > maximum_delta) &  ~(error_vector < -maximum_delta))
        x_k = x_k_new

        if (condition_R_C and condition_U_C):
            break

    Result_from_dcop = x_k

    solution_vector = np.zeros([n, len(steps) + 1])
    solution_vector[:, 0, np.newaxis] = Result_from_dcop

    for i,step in enumerate(steps):

        x_k = solution_vector[:, i, np.newaxis]  # initial guess: last step

        while True:
            # iterate to solve Gx+Hg(x)=s
            #                   j_f . x_k_new = s_k  , s_k = f(x_k)
            j_f, V, s_k = matrix_formulation_tran_non_linear(elements = Circuit_Matrix,
                                                             time_step = time_step,
                                                             last_step_result=solution_vector[:, i, np.newaxis],
                                                             last_iter_results=x_k,
                                                             time = step)

            x_k_new = Solve_real_Linear_Matrix(j_f, s_k)

            # the residue criterion
            f_x_k = j_f @ x_k - s_k
            I_max = max(s_k[:n_nets, 0])
            mag_f_x_k = np.sqrt(np.sum(f_x_k ** 2))
            condition_R_C = (mag_f_x_k < relative_tolerance * abs(I_max) + current_absolute_tolerance)
            # the update criterion
            error_vector = x_k_new - x_k
            error_mag = np.sqrt(np.sum(error_vector ** 2))
            max_nominal_solution = max(max(x_k), max(x_k_new))
            condition_U_C = (error_mag < relative_tolerance * abs(max_nominal_solution) + current_absolute_tolerance)
            # maximum_delta = 0.05  # 2VT = 0.0258 * 2 ~= 0.05
            # todo : make the damping criterion
            # x_k = x_k + maximum_delta * (error_vector > maximum_delta)
            # x_k = x_k - maximum_delta * (error_vector < -maximum_delta)
            # x_k += error_vector * (~(error_vector > maximum_delta) &  ~(error_vector < -maximum_delta))
            x_k = x_k_new

            if (condition_R_C and condition_U_C):
                break

        solution_vector[:, i + 1 , np.newaxis] = x_k

    Result = Divide_Result_Matrix(solution_vector[:,1:], V)
    #steps = np.hstack((np.array([0]) , steps))
    Plot_Output(Circuit_Matrix['plot_name'], steps, Result)

