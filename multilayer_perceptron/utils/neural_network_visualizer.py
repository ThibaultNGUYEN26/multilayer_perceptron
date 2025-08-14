import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class NeuralNetworkVisualizer:
    def __init__(self, layer_sizes, figsize=(16, 10), activation_threshold=0.6, ring_radius=0.10, disc_radius=0.08) -> None:
        """
        Initialize the visualizer.

        Args:
            layer_sizes (list): List of layer sizes [input, hidden1, hidden2, ..., output]
            figsize (tuple): Figure size
            activation_threshold (float): Threshold for neuron activation color intensity
            ring_radius (float): Radius of the neuron activation ring
            disc_radius (float): Radius of the neuron activation disc
        """
        # Store parameters
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.activation_threshold = activation_threshold
        self.ring_radius = ring_radius
        self.disc_radius = disc_radius

        # Setup matplotlib for interactive mode
        plt.ion()

        # Setup the plot
        self.fig, (self.ax_network, self.ax_metrics) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [2, 1]})
        self.fig.subplots_adjust(left=0.04, right=0.995, bottom=0.06, top=0.92, wspace=0.06)
        self.fig.suptitle('Neural Network Training Visualization', fontsize=16, fontweight='bold')

        # Initialize network visualization
        self.setup_network_plot()

        # Initialize metrics plot
        self.setup_metrics_plot()

        # Storage for metrics
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def setup_network_plot(self) -> None:
        """
        Setup the neural network visualization (round neurons, spaced layers).
        """
        # Clear the axes
        ax = self.ax_network
        ax.clear()

        # Set aspect ratio to equal for round neurons
        max_neurons = max(self.layer_sizes)
        L = self.num_layers

        # Set titles and labels
        ax.set_title('Network Activations (Real-time)')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Neuron')
        ax.grid(True, alpha=0.3)

        # use the subplot's real estate to set spacing
        bbox = ax.get_position()
        fig_w, fig_h = self.fig.get_size_inches()
        subplot_w = bbox.width * fig_w
        subplot_h = bbox.height * fig_h

        # Calculate available space for layers and neurons
        available_w = subplot_w * 0.8
        available_h = subplot_h * 0.8

        # Calculate layer and neuron spacing
        layer_spacing  = available_w  / (L - 1) if L > 1 else available_w
        neuron_spacing = available_h / max(1, max_neurons)

        # Set limits based on spacing
        ax.set_xlim(-layer_spacing * 0.1, (L - 1) * layer_spacing + layer_spacing * 0.1)
        ax.set_ylim(-neuron_spacing * 0.9, max_neurons * neuron_spacing + neuron_spacing * 0.7)

        # layer labels
        layer_names = ['Input'] + [f'Hidden {i+1}' for i in range(L - 2)] + ['Output']
        for i, name in enumerate(layer_names):
            x_label = i * layer_spacing
            ax.text(x_label, max_neurons * neuron_spacing + neuron_spacing * 0.05, name,
                    ha='center', va='bottom', fontweight='bold', fontsize=10)

        # storage
        self.neuron_positions = []
        self.neuron_bases = []   # NEW: opaque base discs (mask lines)
        self.neuron_discs = []   # activation overlay discs
        self.neuron_rings = []   # outline rings
        self.weight_lines = []

        # neuron sizes (keep circles round)
        base_r = min(layer_spacing, neuron_spacing) * 0.15
        ring_r = getattr(self, 'ring_radius', base_r)
        disc_r = getattr(self, 'disc_radius', base_r * 0.8)

        ax_bg = ax.get_facecolor()  # match axes background

        # neurons
        for layer_idx, n_neurons in enumerate(self.layer_sizes):
            layer_positions, layer_bases, layer_discs, layer_rings = [], [], [], []
            y_start = (max_neurons - n_neurons) * neuron_spacing / 2

            for j in range(n_neurons):
                x = layer_idx * layer_spacing
                y = y_start + j * neuron_spacing

                # base (always opaque to hide lines), below overlay & ring
                base = patches.Circle((x, y), disc_r, facecolor=ax_bg,
                                    edgecolor='none', zorder=3)
                ax.add_patch(base)

                # activation overlay (colored only when active)
                disc = patches.Circle((x, y), disc_r, facecolor='none',
                                    edgecolor='none', zorder=4)
                ax.add_patch(disc)

                # top outline ring
                ring = patches.Circle((x, y), ring_r, facecolor='none',
                                    edgecolor='black', linewidth=1.6, zorder=5)
                ax.add_patch(ring)

                layer_positions.append((x, y))
                layer_bases.append(base)
                layer_discs.append(disc)
                layer_rings.append(ring)

            self.neuron_positions.append(layer_positions)
            self.neuron_bases.append(layer_bases)
            self.neuron_discs.append(layer_discs)
            self.neuron_rings.append(layer_rings)

        # edges (default: faint black, behind everything)
        for layer_idx in range(L - 1):
            lines_this_layer = []
            src_positions = self.neuron_positions[layer_idx]
            dst_positions = self.neuron_positions[layer_idx + 1]

            for i, (x1, y1) in enumerate(src_positions):
                neuron_lines = []
                for j, (x2, y2) in enumerate(dst_positions):
                    line = ax.plot([x1, x2], [y1, y2],
                                color='black', alpha=0.12, linewidth=0.45, zorder=1)[0]
                    neuron_lines.append(line)
                lines_this_layer.append(neuron_lines)

            self.weight_lines.append(lines_this_layer)


    def _layer_acts_1d(self, layer_idx, activations) -> np.ndarray:
        """
        Get 1D activations for a specific layer.

        Args:
            layer_idx (int): Index of the layer to get activations for.
            activations (dict): Current activations from forward propagation.

        Returns:
            np.ndarray: 1D array of activations for the specified layer.
        """
        if layer_idx == 0:
            a = activations.get('A0', np.zeros(self.layer_sizes[0]))
        else:
            a = activations.get(f'A{layer_idx}', np.zeros(self.layer_sizes[layer_idx]))
        if a.ndim > 1:
            a = a[:, 0]
        return a


    def _norm01(self, a) -> np.ndarray:
        """
        Normalize an array to the range [0, 1].

        Args:
            a (np.ndarray): Input array to normalize.

        Returns:
            np.ndarray: Normalized array with values in [0, 1].
        """
        if a.size == 0:
            return a
        lo, hi = float(a.min()), float(a.max())
        if hi > lo:
            return (a - lo) / (hi - lo)
        return np.full_like(a, 0.5, dtype=float)


    def _active_mask(self, layer_idx, activations) -> tuple:
        """
        Create a mask for active neurons in the specified layer.

        Args:
            layer_idx (int): Index of the layer to check.
            activations (dict): Current activations from forward propagation.

        Returns:
            tuple: (mask, normalized activations)
        """
        a = self._layer_acts_1d(layer_idx, activations)
        n = self._norm01(a) if a.size else np.array([])
        mask = (n >= self.activation_threshold) if n.size else np.array([], dtype=bool)
        return mask, n


    def setup_metrics_plot(self) -> None:
        """
        Setup the metrics visualization.
        """
        self.ax_metrics.set_title('Training Metrics (Real-time)')
        self.ax_metrics.set_xlabel('Epoch')
        self.ax_metrics.set_ylabel('Loss / Accuracy')
        self.ax_metrics.grid(True, alpha=0.3)

        # Initialize empty lines
        self.train_loss_line, = self.ax_metrics.plot([], [], 'b-', label='Train Loss', linewidth=2)
        self.val_loss_line, = self.ax_metrics.plot([], [], 'r-', label='Val Loss', linewidth=2)
        self.train_acc_line, = self.ax_metrics.plot([], [], 'g-', label='Train Acc', linewidth=2)
        self.val_acc_line, = self.ax_metrics.plot([], [], 'm-', label='Val Acc', linewidth=2)

        self.ax_metrics.legend()


    def update_neuron_activations(self, activations) -> None:
        """
        Fill overlay discs only for active neurons; keep a solid base below to mask lines.

        Args:
            activations (dict): Current activations from forward propagation.
        """
        # Ensure base discs always match current axes background
        ax_bg = self.ax_network.get_facecolor()

        for layer_idx in range(len(self.neuron_discs)):
            # keep bases opaque so lines never bleed through
            for base in self.neuron_bases[layer_idx]:
                base.set_facecolor(ax_bg)
                base.set_edgecolor('none')
                base.set_zorder(3)

            # activation mask
            a = (activations.get('A0') if layer_idx == 0
                else activations.get(f'A{layer_idx}', np.zeros(self.layer_sizes[layer_idx])))
            if a is None:
                a = np.zeros(self.layer_sizes[layer_idx])

            if a.ndim > 1:
                a = a[:, 0]

            if a.size:
                lo, hi = float(a.min()), float(a.max())
                norm = (a - lo) / (hi - lo) if hi > lo else np.full_like(a, 0.5, dtype=float)
            else:
                norm = np.array([])

            for i in range(min(len(norm), len(self.neuron_discs[layer_idx]))):
                disc = self.neuron_discs[layer_idx][i]
                ring = self.neuron_rings[layer_idx][i]
                val = float(norm[i])

                if val >= self.activation_threshold:
                    disc.set_facecolor(plt.cm.RdYlBu_r(val))
                    disc.set_edgecolor('black')
                    disc.set_alpha(0.95)
                    disc.set_zorder(4)
                else:
                    # hide overlay; base stays visible and masks edges
                    disc.set_facecolor('none')
                    disc.set_edgecolor('none')
                    disc.set_alpha(1.0)
                    disc.set_zorder(4)

                ring.set_linewidth(1.2 + 1.0 * val)
                ring.set_zorder(5)


    def update_weight_connections(self, parameters, activations) -> None:
        """
        Keep edges black by default; color an edge only if both endpoints are active.
        Edge intensity/width scales with |weight| and both neurons' activations.

        Args:
            parameters (dict): Current weights and biases of the neural network.
            activations (dict): Current activations from forward propagation.
        """
        for layer_idx in range(len(self.weight_lines)):
            W_key = f'W{layer_idx + 1}'
            if W_key not in parameters:
                continue
            W = parameters[W_key]  # shape: (n_dst, n_src)
            if W.size == 0:
                continue

            src_active, src_norm = self._active_mask(layer_idx, activations)
            dst_active, dst_norm = self._active_mask(layer_idx + 1, activations)

            max_abs = np.max(np.abs(W)) + 1e-12

            for src_i in range(len(self.weight_lines[layer_idx])):
                for dst_j in range(len(self.weight_lines[layer_idx][src_i])):
                    line = self.weight_lines[layer_idx][src_i][dst_j]

                    # safety if line grid exceeds W (shouldn't normally)
                    if dst_j >= W.shape[0] or src_i >= W.shape[1]:
                        line.set_color('black'); line.set_alpha(0.12); line.set_linewidth(0.45)
                        continue

                    if src_active.size and dst_active.size and src_active[src_i] and dst_active[dst_j]:
                        w = W[dst_j, src_i]  # note: (dst, src)
                        strength = (abs(w) / max_abs) * float(src_norm[src_i] * dst_norm[dst_j])
                        line.set_color('red' if w >= 0 else 'blue')
                        line.set_alpha(0.20 + 0.80 * strength)
                        line.set_linewidth(0.5 + 1.5 * strength)
                    else:
                        line.set_color('black')
                        line.set_alpha(0.12)
                        line.set_linewidth(0.45)


    def update_metrics(self, epoch, train_loss, val_loss, train_acc, val_acc) -> None:
        """
        Update training metrics plot.

        Args:
            epoch (int): Current training epoch.
            train_loss (float): Training loss for the current epoch.
            val_loss (float): Validation loss for the current epoch.
            train_acc (float): Training accuracy for the current epoch.
            val_acc (float): Validation accuracy for the current epoch.
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)

        # Update lines
        self.train_loss_line.set_data(self.epochs, self.train_losses)
        self.val_loss_line.set_data(self.epochs, self.val_losses)
        self.train_acc_line.set_data(self.epochs, self.train_accs)
        self.val_acc_line.set_data(self.epochs, self.val_accs)

        # Rescale axes
        if self.epochs:
            self.ax_metrics.set_xlim(0, max(self.epochs) + 1)
            all_losses = self.train_losses + self.val_losses
            all_accs = self.train_accs + self.val_accs
            if all_losses and all_accs:
                self.ax_metrics.set_ylim(
                    min(min(all_losses), min(all_accs)) - 0.1,
                    max(max(all_losses), max(all_accs)) + 0.1
                )


    def update_visualization(self, epoch, activations, parameters, train_loss, val_loss, train_acc, val_acc) -> None:
        """
        Update the entire visualization.

        Args:
            epoch (int): Current training epoch.
            activations (dict): Current activations from forward propagation.
            parameters (dict): Current weights and biases of the neural network.
            train_loss (float): Training loss for the current epoch.
            val_loss (float): Validation loss for the current epoch.
            train_acc (float): Training accuracy for the current epoch.
            val_acc (float): Validation accuracy for the current epoch.
        """
        try:
            self.update_neuron_activations(activations)
            self.update_weight_connections(parameters, activations)  # <-- pass activations
            self.update_metrics(epoch, train_loss, val_loss, train_acc, val_acc)

            self.ax_network.set_title(f'Network Activations (Epoch {epoch})')
            self.ax_metrics.set_title(f'Training Metrics - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)
        except Exception as e:
            print(f"Visualization update error: {e}")


    def show(self) -> 'NeuralNetworkVisualizer':
        """
        Show the visualization.
        """
        plt.show(block=False)  # Non-blocking show
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        return self  # Return self for method chaining


    def close(self) -> None:
        """
        Close the visualization.
        """
        plt.close(self.fig)
        plt.ioff()
