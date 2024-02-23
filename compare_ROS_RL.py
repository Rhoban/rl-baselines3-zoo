import numpy as np

planner = "ARA" # "ARA" or "AD"
version = "asimo"

reset_dict_all_situations_arr = np.load("reset_dict_all_situations_ROS.npy", allow_pickle=True, fix_imports=False, encoding="latin1")
# nb_steps_all_situations_ROS = np.load("nb_steps_all_situations_ROS.npy", allow_pickle=True, fix_imports=False, encoding="latin1")

nb_steps_situation1_ROS = np.load("nb_steps_situation1_" + planner + "_ROS_" + version + ".npy", allow_pickle=True, fix_imports=False, encoding="latin1")
nb_steps_situation2_ROS = np.load("nb_steps_situation2_" + planner + "_ROS_" + version + ".npy", allow_pickle=True, fix_imports=False, encoding="latin1")
nb_steps_situation3_ROS = np.load("nb_steps_situation3_" + planner + "_ROS_" + version + ".npy", allow_pickle=True, fix_imports=False, encoding="latin1")

nb_steps_all_situations_RL = np.load("nb_steps_all_situations_RL.npy", allow_pickle=False, fix_imports=False)
nb_steps_all_situations_ROS = np.concatenate((nb_steps_situation1_ROS, nb_steps_situation2_ROS, nb_steps_situation3_ROS))

for nb_situation in range(0, 3):
    reset_dict_arr = reset_dict_all_situations_arr[nb_situation]
    nb_steps_arr_ROS = nb_steps_all_situations_ROS[nb_situation]
    nb_steps_arr_RL = nb_steps_all_situations_RL[nb_situation]

    print(f"\n-----Situation: {nb_situation+1} with obstacle radius = {reset_dict_arr[0]['obstacle_radius']}-----")
    diff_nb_steps = nb_steps_arr_RL - nb_steps_arr_ROS

    nb_positifs = np.sum(diff_nb_steps > 0)
    pourcentage = (nb_positifs / len(diff_nb_steps)) * 100
    
    print(f"Mean number of steps ROS: {np.mean(nb_steps_arr_ROS)}")
    print(f"Mean number of steps RL: {np.mean(nb_steps_arr_RL)}")
    print(f"Mean: {np.mean(diff_nb_steps)}")
    print(f"Pourcentage RL better case: {100-pourcentage}%")
    print(f"Nb steps difference: {diff_nb_steps}")

    mean_relative_error = ((nb_steps_arr_RL - nb_steps_arr_ROS)/nb_steps_arr_ROS)*100