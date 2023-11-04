import visdom

vis = visdom.Visdom(server='http://localhost', port=8097)


def plot_loss(iteration, loss_D, loss_G):
    vis.line(
        X=[iteration],
        Y=[loss_D],
        win='Loss D',
        update='append' if iteration > 0 else None,
        opts=dict(title='Discriminator Loss', xlabel='Iteration', ylabel='Loss'),
        )
    vis.line(
        X=[iteration],
        Y=[loss_G],
        win='Loss G',
        update='append' if iteration > 0 else None,
        opts=dict(title='Generator Loss', xlabel='Iteration', ylabel='Loss'),
        )