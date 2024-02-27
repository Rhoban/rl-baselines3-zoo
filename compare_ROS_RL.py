import numpy as np

planner = "ARA" # "ARA" or "AD"
version = "asimo" # "nao" or "asimo"

reset_dict_all_situations_arr = np.load("reset_dict_all_situations_ROS_500.npy", allow_pickle=True, fix_imports=False, encoding="latin1")
nb_tests = np.shape(reset_dict_all_situations_arr)[1]

nb_steps_situation1_ROS = np.load("nb_steps_situation1_" + planner + "_ROS_" + version + "_" + str(nb_tests) + ".npy", allow_pickle=True, fix_imports=False, encoding="latin1")
nb_steps_situation2_ROS = np.load("nb_steps_situation2_" + planner + "_ROS_" + version + "_" + str(nb_tests) + ".npy", allow_pickle=True, fix_imports=False, encoding="latin1")
nb_steps_situation3_ROS = np.load("nb_steps_situation3_" + planner + "_ROS_" + version + "_" + str(nb_tests) + ".npy", allow_pickle=True, fix_imports=False, encoding="latin1")

nb_steps_all_situations_RL = np.load("nb_steps_all_situations_RL_500.npy", allow_pickle=False, fix_imports=False)
nb_steps_all_situations_ROS = np.concatenate((nb_steps_situation1_ROS, nb_steps_situation2_ROS, nb_steps_situation3_ROS))

for nb_situation in range(1, 4):
    reset_dict_arr = reset_dict_all_situations_arr[nb_situation-1]
    nb_steps_arr_ROS = nb_steps_all_situations_ROS[nb_situation-1]
    nb_steps_arr_RL = nb_steps_all_situations_RL[nb_situation-1]

    print(f"\n-----Situation: {nb_situation} with obstacle radius = {reset_dict_arr[0]['obstacle_radius']} Nb. of tests : {nb_tests}-----")
    diff_nb_steps = nb_steps_arr_RL - nb_steps_arr_ROS

    nb_positifs = np.sum(diff_nb_steps > 0)
    pourcentage = (nb_positifs / len(diff_nb_steps)) * 100
    
    print(f"Mean number of steps ROS: {np.mean(nb_steps_arr_ROS)}, SD: {np.round(np.std(nb_steps_arr_ROS),2)}")
    print(f"Mean number of steps RL: {np.mean(nb_steps_arr_RL)}, SD: {np.round(np.std(nb_steps_arr_RL),2)}")
    print(f"Mean: {np.mean(diff_nb_steps)}, SD: {np.round(np.std(diff_nb_steps),2)}")
    print(f"Pourcentage RL better case: {100-pourcentage}%")
    # print(f"Nb steps difference: {diff_nb_steps}")

    abs_mean_relative_ROS_error = ((np.abs(nb_steps_arr_RL - nb_steps_arr_ROS))/nb_steps_arr_ROS)*100
    abs_mean_relative_RL_error = (np.abs((nb_steps_arr_ROS - nb_steps_arr_RL))/nb_steps_arr_RL)*100
    mean_relative_ROS_error = ((nb_steps_arr_RL - nb_steps_arr_ROS)/nb_steps_arr_ROS)*100
    mean_relative_RL_error = ((nb_steps_arr_ROS - nb_steps_arr_RL)/nb_steps_arr_RL)*100

    print(f"Mean relative to ROS error: {np.round(np.mean(mean_relative_ROS_error),2)}%")
    print(f"Mean relative to RL error: {np.round(np.mean(mean_relative_RL_error),2)}%")
    print(f"Abs. Mean relative to ROS error: {np.round(np.mean(abs_mean_relative_ROS_error),2)}%")
    print(f"Abs. Mean relative to RL error: {np.round(np.mean(abs_mean_relative_RL_error),2)}%")