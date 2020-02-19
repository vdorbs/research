class Visual:
    def boundary(self, N):
        raise NotImplementedError

    def plot_boundary(self, ax, N):
        return ax.plot(*self.boundary(N).T, 'k', linewidth=3)
