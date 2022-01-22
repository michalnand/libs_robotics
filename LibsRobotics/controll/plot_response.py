import matplotlib.pyplot as plt


def plot_output(t_trajectory, u_trajectory, y_trajectory, u_name, y_names, output_file_name):

    t_trajectory = t_trajectory.detach().to("cpu").numpy()
    u_trajectory = u_trajectory.detach().to("cpu").numpy()
    y_trajectory = y_trajectory.detach().to("cpu").numpy()


    u_mean  = u_trajectory.mean(axis=1)[:, 0]
    u_min   = u_trajectory.min(axis=1)[:, 0]
    u_max   = u_trajectory.max(axis=1)[:, 0]

    y_mean  = y_trajectory.mean(axis=1)
    y_min   = y_trajectory.min(axis=1)
    y_max   = y_trajectory.max(axis=1)

   
    plt.clf()
    fig, axs = plt.subplots(1 + y_mean.shape[1], 1, figsize=(5,6))

    axs[0].fill_between(t_trajectory, u_min, u_max, facecolor='red', alpha=0.3)
    axs[0].plot(t_trajectory, u_mean, color='red')
    axs[0].set_xlabel("time [s]")
    axs[0].set_ylabel(u_name)
    axs[0].grid(True)

    for i in range(y_mean.shape[1]):
        axs[1 + i].fill_between(t_trajectory, y_min[:, i], y_max[:, i], facecolor='deepskyblue', alpha=0.3)
        axs[1 + i].plot(t_trajectory, y_mean[:, i], color='deepskyblue')
        axs[1 + i].set_xlabel("time [s]")
        axs[1 + i].set_ylabel(y_names[i])
        axs[1 + i].grid(True)

    fig.tight_layout()
    plt.savefig(output_file_name, dpi = 300)
    plt.close()


def plot_controll_output(t_trajectory, u_trajectory, y_req_trajectory, y_test_trajectory, y_names, output_file_name):

    t_trajectory        = t_trajectory.detach().to("cpu").numpy()
    u_trajectory        = u_trajectory.detach().to("cpu").numpy()
    y_req_trajectory    = y_req_trajectory.detach().to("cpu").numpy()
    y_test_trajectory   = y_test_trajectory.detach().to("cpu").numpy()


    u_mean  = u_trajectory.mean(axis=1).mean(axis=1)
    u_min   = u_trajectory.min(axis=1).min(axis=1)
    u_max   = u_trajectory.max(axis=1).max(axis=1)

    y_req_mean  = y_req_trajectory.mean(axis=1)
    y_req_min   = y_req_trajectory.min(axis=1)
    y_req_max   = y_req_trajectory.max(axis=1)

    y_test_mean  = y_test_trajectory.mean(axis=1)
    y_test_min   = y_test_trajectory.min(axis=1)
    y_test_max   = y_test_trajectory.max(axis=1)

 
   
    plt.clf()
    fig, axs = plt.subplots(1 + y_req_mean.shape[1], 1, figsize=(5,6))

    axs[0].fill_between(t_trajectory, u_min, u_max, facecolor='red', alpha=0.3)
    axs[0].plot(t_trajectory, u_mean, color='red')
    axs[0].set_xlabel("time [s]")
    axs[0].set_ylabel("controller output")
    axs[0].grid(True)

    for i in range(y_req_mean.shape[1]):
        axs[1 + i].fill_between(t_trajectory, y_req_min[:, i], y_req_max[:, i], facecolor='red', alpha=0.3)
        axs[1 + i].fill_between(t_trajectory, y_test_min[:, i], y_test_max[:, i], facecolor='deepskyblue', alpha=0.3)

        axs[1 + i].plot(t_trajectory, y_req_mean[:, i], color='red', label="required output")
        axs[1 + i].plot(t_trajectory, y_test_mean[:, i], color='deepskyblue', label="plant output")
        
        axs[1 + i].set_xlabel("time [s]")
        axs[1 + i].set_ylabel(y_names[i])
        axs[1 + i].grid(True)

        axs[1 + i].legend(loc='lower right')

    fig.tight_layout()
    plt.savefig(output_file_name, dpi = 300)
    plt.close()
