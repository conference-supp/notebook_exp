def plot_perf_metrics(ls_rewards, ls_opt_act_flgs, ls_param, param_prefix='epsilon', plt_lib='mlp'):
    if plt_lib == 'mlp':
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        
        for i, eps in enumerate(ls_param):
            axes[0].plot(np.mean(ls_rewards[i], axis=0), label='param = {}'.format(eps))
            axes[1].plot(np.mean(ls_opt_act_flgs[i], axis=0), label='param = {}'.format(eps))
        
        axes[0].set_title('Average Reward')
        axes[0].set_xlabel('Steps')
        axes[0].set_ylabel('Average Reward')
        axes[0].legend()
        
        axes[1].set_title('% Optimal Action')
        axes[1].set_xlabel('Steps')
        axes[1].set_ylabel('% Optimal Action')
        axes[1].legend()
        
        plt.show()
    else:
        # Create subplots with 2 rows and 1 column
        fig = make_subplots(rows=2, cols=1)

        for i, eps in enumerate(ls_param):
            
            # Add traces to the first subplot for rewards
            fig.add_trace(go.Scatter(x=np.arange(ls_rewards[0].shape[1]),
                                    y=np.mean(ls_rewards[i], axis=0),
                                    mode='lines',
                                    name='{} = {}'.format(param_prefix, eps),
                                    line=dict(color=COLOR_LIST[i])), row=1, col=1)

            # Add traces to the second subplot for optimal action rate
            fig.add_trace(go.Scatter(x=np.arange(ls_opt_act_flgs[0].shape[1]),
                                    y=np.mean(ls_opt_act_flgs[i], axis=0),
                                    mode='lines',
                                    # name='param = {}'.format(eps),
                                    line=dict(color=COLOR_LIST[i]),
                                    showlegend=False),
                                    row=2, col=1)
        
        # Add axis titles
        fig.update_xaxes(title_text='Steps', row=1, col=1)
        fig.update_yaxes(title_text='Average Reward', row=1, col=1)
        fig.update_xaxes(title_text='Steps', row=2, col=1)
        fig.update_yaxes(title_text='% Optimal Action', row=2, col=1)

        # Update layout
        fig.update_layout(height=600, width=800, title_text='Subplots')

        # Show the figure
        fig.show()
        