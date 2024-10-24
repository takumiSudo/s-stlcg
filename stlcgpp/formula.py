import torch
from stlcgpp.utils import *

# Expressions
class Expression(torch.nn.Module):
    def __init__(self, name, value):
        super(Expression, self).__init__()
        self.value = value
        self.name = name

    def set_name(self, new_name):
        self.name = new_name

    def set_value(self, new_value):
        self.value = new_value

    def get_name(self):
        return self.name

    def forward(self):
        return self.value

# Predicates

class Predicate(torch.nn.Module):
    def __init__(self, name, predicate: Callable):
        super(Predicate, self).__init__()
        self.name = name
        self.predicate = predicate

    def forward(self, signal: torch.Tensor):
        return self.predicate(signal)




def convert_to_input_values(inputs):
    if not isinstance(inputs, tuple):
        if isinstance(inputs, Expression):
            assert inputs.value is not None, "Input Expression does not have numerical values"
            # if Expression is not time reversed
            return inputs.value
        elif isinstance(inputs, torch.Tensor):
            return inputs
        else:
            raise ValueError("Not a invalid input trace")
    else:
        return (convert_to_input_values(inputs[0]), convert_to_input_values(inputs[1]))



# STL formula
class STLFormula(torch.nn.Module):
    '''
    Class for an STL formula
    NOTE: If Expressions and Predicates are used, then the signals will be reversed if needed. Otherwise, user is responsibile for keeping track.
    '''
    def __init__(self):
        super(STLFormula, self).__init__()
        
    def robustness_trace(self, signal: torch.Tensor, **kwargs):
        """
        Computes the robustness trace of the formula given an input signal.

        Args:
            signal: jnp.array. Expected size [time_dim, state_dim]
            kwargs: Other arguments including time_dim, approx_method, temperature

        Returns:
            robustness_trace: jnp.array of size equal to the input. index=0 along dim=time_dim is the robustness of the last subsignal. index=-1 along dim=time_dim is the robustness of the entire signal.
        """
        # return signal
        raise NotImplementedError("robustness_trace not yet implemented")
    
    def robustness(self, signal: torch.Tensor, **kwargs):
        """
        Computes the robustness value. Extracts the last entry along time_dim of robustness trace.

        Args:
            signal: jnp.array or Expression. Expected size [bs, time_dim, state_dim]
            kwargs: Other arguments including time_dim, approx_method, temperature

        Return: jnp.array, same as input with the time_dim removed.
        """
        return self.forward(signal, **kwargs)[0]
    
    def eval_trace(self, signal: torch.Tensor, **kwargs):
        """
        Boolean of robustness_trace

        Args:
            signal: jnp.array or Expression. Expected size [bs, time_dim, state_dim]
            kwargs: Other arguments including time_dim, approx_method, temperature

        Returns:
            eval_trace: jnp.array of size equal to the input but with True/False. index=0 along dim=time_dim is the robustness of the last subsignal. index=-1 along dim=time_dim is the robustness of the entire signal.
        """

        return self.forward(signal, **kwargs) > 0
    
    def eval(self, signal: torch.Tensor, **kwargs):
        """
        Boolean of robustness

        Args:
            signal: jnp.array or Expression. Expected size [bs, time_dim, state_dim]
            kwargs: Other arguments including time_dim, approx_method, temperature

        Return: jnp.array with True/False, same as input with the time_dim removed.
        """
        return self.robustness(signal, **kwargs) > 0

    def forward(self, signal: torch.Tensor, **kwargs):
        """    
        Evaluates the robustness_trace given the input. The input is converted to the numerical value first.

        See  STLFormula.robustness_trace
        """

        inputs = convert_to_input_values(signal)
        return self.robustness_trace(inputs, **kwargs)

    def _next_function(self):
        """Function to keep track of the subformulas. For visualization purposes"""
        raise NotImplementedError("_next_function not year implemented")

class Identity(STLFormula):
    """ The identity formula. Use in UntilRecurrent"""

    def __init__(self, name='x'):
        super().__init__()
        self.name = name

    def robustness_trace(self, signal: torch.Tensor, **kwargs):
        return signal

    def _next_function(self):
        return []

    def __str__(self):
        return "%s" %self.name


class LessThan(STLFormula):
    def __init__(self, lhs: Union[Predicate, Expression, str], rhs: Union[float, torch.Tensor]):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs

    def robustness_trace(self, signal: Union[torch.Tensor, Expression], **kwargs):
        if isinstance(self.lhs, Predicate):
            return (self.rhs - self.lhs(signal))
        elif isinstance(signal, Expression):
            assert signal.value is not None, "Expression does not have numerical values"
            return (self.rhs - signal.value)
        else:
            return (self.rhs - signal)


    def _next_function(self):
        """ next function is the input subformula. For visualization purposes """
        return [self.lhs, self.rhs]

    def __str__(self):
        lhs_str = self.lhs
        if isinstance(self.lhs, Predicate) or isinstance(self.lhs, Expression):
            lhs_str = self.lhs.name
        return lhs_str + " < " + str(self.rhs)


class GreaterThan(STLFormula):
    def __init__(self, lhs: Union[Predicate, Expression, str], rhs: Union[float, torch.Tensor]):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs

    def robustness_trace(self, signal: Union[torch.Tensor, Expression], **kwargs):
        if isinstance(self.lhs, Predicate):
            return (self.lhs(signal) - self.rhs)
        elif isinstance(signal, Expression):
            assert signal.value is not None, "Expression does not have numerical values"
            return (signal.value - self.rhs)
        else:
            return (signal - self.rhs)


    def _next_function(self):
        """ next function is the input subformula. For visualization purposes """
        return [self.lhs, self.rhs]

    def __str__(self):
        lhs_str = self.lhs
        if isinstance(self.lhs, Predicate) or isinstance(self.lhs, Expression):
            lhs_str = self.lhs.name
        return lhs_str + " > " + str(self.rhs)


class Equal(STLFormula):
    def __init__(self, lhs: Union[Predicate, Expression, str], rhs: Union[float, torch.Tensor]):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs

    def robustness_trace(self, signal: Union[torch.Tensor, Expression], **kwargs):
        if isinstance(self.lhs, Predicate):
            return -torch.abs((self.lhs(signal) - self.rhs))
        elif isinstance(signal, Expression):
            assert signal.value is not None, "Expression does not have numerical values"
            return -torch.abs(signal.value - self.rhs)
        else:
            return -torch.abs(signal - self.rhs)


    def _next_function(self):
        """ next function is the input subformula. For visualization purposes """
        return [self.lhs, self.rhs]

    def __str__(self):
        lhs_str = self.lhs
        if isinstance(self.lhs, Predicate) or isinstance(self.lhs, Expression):
            lhs_str = self.lhs.name
        return lhs_str + " == " + str(self.rhs)



class Negation(STLFormula):
    def __init__(self, subformula: STLFormula):
        super().__init__()
        self.subformula = subformula

    def robustness_trace(self, signal: Union[torch.Tensor, Expression], **kwargs):
        return -self.subformula(signal, **kwargs)

    def _next_function(self):
        """ next function is the input subformula. For visualization purposes """
        return [self.subformula]

    def __str__(self):
        return "¬(" + str(self.subformula) + ")"



class And(STLFormula):
    """
    The And STL formula ∧ (subformula1 ∧ subformula2)
    Args:
        subformula1: subformula for lhs of the And operation
        subformula2: subformula for rhs of the And operation
    """

    def __init__(self, subformula1, subformula2):
        super().__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2

    def robustness_trace(self, inputs, **kwargs):
        """
        Computing robustness trace of subformula1 ∧ subformula2  min(subformula1(input1), subformula2(input2))

        Args:
            inputs: input signal for the formula. If using Expressions to define the formula, then inputs a tuple of signals corresponding to each subformula. Each element of the tuple could also be a tuple if the corresponding subformula requires multiple inputs (e.g, ϕ₁(x) ∧ (ϕ₂(y) ∧ ϕ₃(z)) would have inputs=(x, (y,z))). If using Predicates to define the formula, then inputs is just a single jnp.array. Not need for different signals for each subformula. Expected signal is size [batch_size, time_dim, x_dim]
            kwargs: Other arguments including time_dim, approx_method, temperature

        Returns:
            robustness_trace: jnp.array. Same size as signal.
        """
        xx = separate_and(self, inputs, **kwargs)
        return minish(xx, dim=-1, keepdim=False, **kwargs)                                         # [batch_size, time_dim, ...]

    def _next_function(self):
        """ next function is the input subformulas. For visualization purposes """
        return [self.subformula1, self.subformula2]

    def __str__(self):
        return "(" + str(self.subformula1) + ") ∧ (" + str(self.subformula2) + ")"


class Or(STLFormula):
    """
    The Or STL formula ∧ (subformula1 ∧ subformula2)
    Args:
        subformula1: subformula for lhs of the Or operation
        subformula2: subformula for rhs of the Or operation
    """

    def __init__(self, subformula1, subformula2):
        super().__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2

    def robustness_trace(self, inputs, **kwargs):
        """
        Computing robustness trace of subformula1 ∧ subformula2  min(subformula1(input1), subformula2(input2))

        Args:
            inputs: input signal for the formula. If using Expressions to define the formula, then inputs a tuple of signals corresponding to each subformula. Each element of the tuple could also be a tuple if the corresponding subformula requires multiple inputs (e.g, ϕ₁(x) ∧ (ϕ₂(y) ∧ ϕ₃(z)) would have inputs=(x, (y,z))). If using Predicates to define the formula, then inputs is just a single jnp.array. Not need for different signals for each subformula. Expected signal is size [batch_size, time_dim, x_dim]
            kwargs: Other arguments including time_dim, approx_method, temperature

        Returns:
            robustness_trace: jnp.array. Same size as signal.
        """
        xx = separate_or(self, inputs, **kwargs)
        return maxish(xx, dim=-1, keepdim=False, **kwargs)                                         # [batch_size, time_dim, ...]

    def _next_function(self):
        """ next function is the input subformulas. For visualization purposes """
        return [self.subformula1, self.subformula2]

    def __str__(self):
        return "(" + str(self.subformula1) + ") ∨ (" + str(self.subformula2) + ")"


class Implies(STLFormula):
    """
    The Implies STL formula ⇒. subformula1 ⇒ subformula2
    Args:
        subformula1: subformula for lhs of the Implies operation
        subformula2: subformula for rhs of the Implies operation
    """
    def __init__(self, subformula1, subformula2):
        super().__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2

    def robustness_trace(self, trace, **kwargs):
        """
        Computing robustness trace of subformula1 ⇒ subformula2    max(-subformula1(input1), subformula2(input2))

        Args:
            inputs: input signal for the formula. If using Expressions to define the formula, then inputs a tuple of signals corresponding to each subformula. Each element of the tuple could also be a tuple if the corresponding subformula requires multiple inputs (e.g, ϕ₁(x) ∨ (ϕ₂(y) ∨ ϕ₃(z)) would have inputs=(x, (y,z))). If using Predicates to define the formula, then inputs is just a single jnp.array. Not need for different signals for each subformula. Expected signal is size [batch_size, time_dim, x_dim]
            kwargs: Other arguments including time_dim, approx_method, temperature

        Returns:
            robustness_trace: jnp.array. Same size as signal.
        """
        if isinstance(trace, tuple):
            trace1, trace2 = trace
            signal1 = self.subformula1(trace1, **kwargs)
            signal2 = self.subformula2(trace2, **kwargs)
        else:
            signal1 = self.subformula1(trace, **kwargs)
            signal2 = self.subformula2(trace, **kwargs)
        xx = torch.stack([-signal1, signal2], dim=-1)      # [time_dim, ..., 2]
        return maxish(xx, dim=-1, keepdim=False, **kwargs)   # [time_dim, ...]


    def _next_function(self):
        """ next function is the input subformulas. For visualization purposes """
        return [self.subformula1, self.subformula2]

    def __str__(self):
        return "(" + str(self.subformula1) + ") ⇒ (" + str(self.subformula2) + ")"


class Eventually(STLFormula):
    def __init__(self, subformula, interval=None):
        super().__init__()

        self.interval = interval
        self.subformula = subformula
        self._interval = [0, torch.inf] if self.interval is None else self.interval

    def robustness_trace(self, signal, padding="last", large_number=1E9, **kwargs):
        device = signal.device
        time_dim = 0  # assuming signal is [time_dim,...]
        signal = self.subformula(signal, padding=padding, large_number=large_number, **kwargs)
        T = signal.shape[time_dim]
        mask_value = -large_number
        if self.interval is None:
            interval = [0,T-1]
        else:
            interval = self.interval
        signal_matrix = signal.reshape([T,1]) @ torch.ones([1,T], device=device)
        if padding == "last":
            pad_value = signal[-1]
        elif padding == "mean":
            pad_value = signal.mean(time_dim)
        else:
            pad_value = padding
        signal_pad = torch.ones([interval[1]+1, T], device=device) * pad_value
        signal_padded = torch.cat([signal_matrix, signal_pad], dim=time_dim)
        subsignal_mask = torch.tril(torch.ones([T + interval[1]+1,T], device=device))
        time_interval_mask = torch.triu(torch.ones([T + interval[1]+1,T], device=device), -interval[-1]) * torch.tril(torch.ones([T + interval[1]+1,T], device=device), -interval[0])
        masked_signal_matrix = torch.where((time_interval_mask * subsignal_mask) == 1., signal_padded, mask_value)
        return maxish(masked_signal_matrix, dim=time_dim, keepdim=False, **kwargs)

    def _next_function(self):
        """ next function is the input subformula. For visualization purposes """
        return [self.subformula]

    def __str__(self):
        return "♢ " + str(self._interval) + "( " + str(self.subformula) + " )"


class Always(STLFormula):
    def __init__(self, subformula, interval=None):
        super().__init__()

        self.interval = interval
        self.subformula = subformula
        self._interval = [0, torch.inf] if self.interval is None else self.interval

    def robustness_trace(self, signal, padding="last", large_number=1E9, **kwargs):
        device = signal.device
        time_dim = 0  # assuming signal is [time_dim,...]
        signal = self.subformula(signal, padding=padding, large_number=large_number, **kwargs)
        T = signal.shape[time_dim]
        mask_value = large_number
        if self.interval is None:
            interval = [0,T-1]
        else:
            interval = self.interval
        signal_matrix = signal.reshape([T,1]) @ torch.ones([1,T], device=device)
        if padding == "last":
            pad_value = signal[-1]
        elif padding == "mean":
            pad_value = signal.mean(time_dim)
        else:
            pad_value = padding
        signal_pad = torch.ones([interval[1]+1, T], device=device) * pad_value
        signal_padded = torch.cat([signal_matrix, signal_pad], dim=time_dim)
        subsignal_mask = torch.tril(torch.ones([T + interval[1]+1,T], device=device))
        time_interval_mask = torch.triu(torch.ones([T + interval[1]+1,T], device=device), -interval[-1]) * torch.tril(torch.ones([T + interval[1]+1,T], device=device), -interval[0])
        masked_signal_matrix = torch.where((time_interval_mask * subsignal_mask) == 1., signal_padded, mask_value)
        return minish(masked_signal_matrix, dim=time_dim, keepdim=False, **kwargs)

    def _next_function(self):
        """ next function is the input subformula. For visualization purposes """
        return [self.subformula]

    def __str__(self):
        return "◻ " + str(self._interval) + "( " + str(self.subformula) + " )"


class Until(STLFormula):
    def __init__(self, subformula1, subformula2, interval=None):
        super().__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        self.interval = interval
        self._interval = [0, torch.inf] if self.interval is None else self.interval


    def robustness_trace(self, signal, padding="last", large_number=1E9, **kwargs):
        device =signal.device
        time_dim = 0  # assuming signal is [time_dim,...]
        if isinstance(signal, tuple):
            signal1, signal2 = signal
            assert signal1.shape[time_dim] == signal2.shape[time_dim]
            signal1 = self.subformula1(signal1, padding=padding, large_number=large_number, **kwargs)
            signal2 = self.subformula2(signal2, padding=padding, large_number=large_number, **kwargs)
            T = signal1.shape[time_dim]
        else:
            signal1 = self.subformula1(signal, padding=padding, large_number=large_number, **kwargs)
            signal2 = self.subformula2(signal, padding=padding, large_number=large_number, **kwargs)
            T = signal.shape[time_dim]

        mask_value = large_number
        if self.interval is None:
            interval = [0,T-1]
        else:
            interval = self.interval
        signal1_matrix = signal1.reshape([T,1]) @ torch.ones([1,T], device=device)
        signal2_matrix = signal2.reshape([T,1]) @ torch.ones([1,T], device=device)
        if padding == "last":
            signal1_pad = torch.ones([interval[1]+1, T], device=device) * signal1[-1]
            signal2_pad = torch.ones([interval[1]+1, T], device=device) * signal2[-1]
        elif padding == "mean":
            signal1_pad = torch.ones([interval[1]+1, T], device=device) * signal1.mean(time_dim)
            signal2_pad = torch.ones([interval[1]+1, T], device=device) * signal2.mean(time_dim)
        else:
            signal1_pad = torch.ones([interval[1]+1, T], device=device) * padding
            signal2_pad = torch.ones([interval[1]+1, T], device=device) * padding

        signal1_padded = torch.cat([signal1_matrix, signal1_pad], dim=time_dim)
        signal2_padded = torch.cat([signal2_matrix, signal2_pad], dim=time_dim)

        phi1_mask = torch.stack([torch.triu(torch.ones([T + interval[1]+1,T]), -end_idx) * torch.tril(torch.ones([T + interval[1]+1,T])) for end_idx in range(interval[0], interval[-1]+1)], 0)
        phi2_mask = torch.stack([torch.triu(torch.ones([T + interval[1]+1,T]), -end_idx) * torch.tril(torch.ones([T + interval[1]+1,T]), -end_idx) for end_idx in range(interval[0], interval[-1]+1)], 0)
        phi1_masked_signal = torch.stack([torch.where(m1==1.0, signal1_padded, mask_value) for m1 in phi1_mask], 0)
        phi2_masked_signal = torch.stack([torch.where(m2==1.0, signal2_padded, mask_value) for m2 in phi2_mask], 0)
        return maxish(torch.stack([minish(torch.stack([minish(s1, dim=0, keepdim=False), minish(s2, dim=0, keepdim=False)], dim=0), dim=0, keepdim=False) for (s1, s2) in zip(phi1_masked_signal, phi2_masked_signal)], dim=0), dim=0, keepdim=False)

    def _next_function(self):
        """ next function is the input subformula. For visualization purposes """
        return [self.subformula1, self.subformula2]

    def __str__(self):
        return  "(" + str(self.subformula1) + ")" + " U " + str(self._interval) + "(" + str(self.subformula2) + ")"


class TemporalOperator(STLFormula):
    """
    Class to compute Eventually and Always. This builds a recurrent cell to perform dynamic programming

    Args:
        subformula: The subformula that the temporal operator is applied to.
        interval: The time interval that the temporal operator operates on. Default: None which means [0, torch.inf]. Other options car be: [a, b] (b < torch.inf), [a, torch.inf] (a > 0)

    NOTE: Assume that the interval is describing the INDICES of the desired time interval. The user is responsible for converting the time interval (in time units) into indices (integers) using knowledge of the time step size.
    """
    def __init__(self, subformula, interval=None):
        super().__init__()
        self.subformula = subformula
        self.interval = interval
        self._interval = [0, torch.inf] if self.interval is None else self.interval
        self.hidden_dim = 1 if not self.interval else self.interval[-1]    # hidden_dim=1 if interval is [0, ∞) otherwise hidden_dim=end of interval
        if self.hidden_dim == torch.inf:
            self.hidden_dim = self.interval[0]
        self.steps = 1 if not self.interval else self.interval[-1] - self.interval[0] + 1   # steps=1 if interval is [0, ∞) otherwise steps=length of interval
        self.operation = None



    def _initialize_hidden_state(self, signal):
        """
        Compute the initial hidden state.

        Args:
            signal: the input signal. Expected size [time_dim,]

        Returns:
            h0: initial hidden state is [hidden_dim,]

        Notes:
        Initializing the hidden state requires padding on the signal. Currently, the default is to extend the last value.
        TODO: have option on this padding

        """
        device = signal.device

        # Matrices that shift a vector and add a new entry at the end.
        self.M = torch.diag(torch.ones(self.hidden_dim-1, device=device), diagonal=1)
        self.b = torch.zeros(self.hidden_dim, device=device)
        self.b[-1] = 1.0

        # Case 1, 2, 4
        # TODO: make this less hard-coded. Assumes signal is [bs, time_dim, signal_dim], and already reversed
        # pads with the signal value at the last time step.
        h0 = torch.ones([self.hidden_dim, *signal.shape[1:]], device=device) * signal[:1]

        # Case 3: if self.interval is [a, torch.inf), then the hidden state is a tuple (like in an LSTM)
        if (self._interval[1] == torch.inf) & (self._interval[0] > 0):
            c0 = signal[:1]
            return (c0, h0)
        return h0

    def cell(self, x, hidden_state, **kwargs):
        """
        This function describes the operation that takes place at each recurrent step.
        Args:
            x: the input state at time t [batch_size, 1, ...]
            hidden_state: the hidden state. It is either a tensor, or a tuple of tensors, depending on the interval chosen and other arguments. Generally, the hidden state is of size [batch_size, hidden_dim,...]

        Return:
            output and next hidden_state
        """
        raise NotImplementedError("cell is not implemented")

    def _cell(self, x, hidden_state, operation, **kwargs):
        time_dim = 0
        # Case 1, interval = [0, inf]
        if self.interval is None:
            input_ = torch.concatenate([hidden_state, x], axis=time_dim)                # [rnn_dim+1,]
            output = operation(input_, dim=time_dim, **kwargs)       # [1,]
            return output, output

        # Case 3: self.interval is [a, np.inf)
        if (self._interval[1] == torch.inf) & (self._interval[0] > 0):
            c, h = hidden_state
            ch = torch.concatenate([c, h[:1]], axis=time_dim)                             # [2,]
            output = operation(ch, dim=time_dim, **kwargs)               # [1,]
            hidden_state_ = (output, self.M @ h + self.b * x)

        # Case 2 and 4: self.interval is [a, b]
        else:
            hidden_state_ = self.M @ hidden_state + self.b * x
            hx = torch.concatenate([hidden_state, x], axis=time_dim)                             # [rnn_dim+1,]
            input_ = hx[:self.steps]                               # [self.steps,]
            output = operation(input_, dim=time_dim, **kwargs)               # [1,]
        return output, hidden_state_


    def _run_cell(self, signal, **kwargs):
        """
        Function to run a signal through a cell T times, where T is the length of the signal in the time dimension

        Args:
            signal: input signal, size = [time_dim,]
            time_dim: axis corresponding to time_dim. Default: 0
            kwargs: Other arguments including time_dim, approx_method, temperature

        Return:
            outputs: list of outputs
            states: list of hidden_states
        """
        time_dim = 0  # assuming signal is [time_dim,...]
        hidden_state = self._initialize_hidden_state(signal)                               # [hidden_dim]
        outputs = []
        states = []

        signal_split = torch.split(signal, 1, time_dim)    # list of x at each time step
        for i in range(signal.shape[time_dim]):
            o, hidden_state = self.cell(signal_split[i], hidden_state, **kwargs)
            outputs.append(o)
            states.append(hidden_state)
        return outputs, states


    def robustness_trace(self, signal, **kwargs):
        """
        Function to compute robustness trace of a temporal STL formula
        First, compute the robustness trace of the subformula, and use that as the input for the recurrent computation

        Args:
            signal: input signal, size = [bs, time_dim, ...]
            time_dim: axis corresponding to time_dim. Default: 1
            kwargs: Other arguments including time_dim, approx_method, temperature

        Returns:
            robustness_trace: torch.array. Same size as signal.
        """
        time_dim = 0  # assuming signal is [time_dim,...]
        trace = self.subformula(signal, **kwargs)
        outputs, _ = self._run_cell(trace, **kwargs)
        return torch.concatenate(outputs, axis=time_dim)                     # [time_dim, ]

    def _next_function(self):
        """ next function is the input subformula. For visualization purposes """
        return [self.subformula]


class AlwaysRecurrent(TemporalOperator):
    """
    The Always STL formula □_[a,b] subformula
    The robustness value is the minimum value of the input trace over a prespecified time interval

    Args:
        subformula: subformula that the Always operation is applied on
        interval: time interval [a,b] where a, b are indices along the time dimension. It is up to the user to keep track of what the timestep size is.
    """
    def __init__(self, subformula, interval=None):
        super().__init__(subformula=subformula, interval=interval)

    def cell(self, x, hidden_state, **kwargs):
        return self._cell(x, hidden_state, minish, **kwargs)

    def __str__(self):
        return "◻ " + str(self._interval) + "( " + str(self.subformula) + " )"


class EventuallyRecurrent(TemporalOperator):
    """
    The Eventually STL formula □_[a,b] subformula
    The robustness value is the minimum value of the input trace over a prespecified time interval

    Args:
        subformula: subformula that the Eventually operation is applied on
        interval: time interval [a,b] where a, b are indices along the time dimension. It is up to the user to keep track of what the timestep size is.
    """
    def __init__(self, subformula, interval=None):
        super().__init__(subformula=subformula, interval=interval)

    def cell(self, x, hidden_state, **kwargs):
        return self._cell(x, hidden_state, maxish, **kwargs)

    def __str__(self):
        return "♢ " + str(self._interval) + "( " + str(self.subformula) + " )"


class UntilRecurrent(STLFormula):
    """
    The Until STL operator U. Subformula1 U_[a,b] subformula2
    Arg:
        subformula1: subformula for lhs of the Until operation
        subformula2: subformula for rhs of the Until operation
        interval: time interval [a,b] where a, b are indices along the time dimension. It is up to the user to keep track of what the timestep is.
        overlap: If overlap=True, then the last time step that ϕ is true, ψ starts being true. That is, sₜ ⊧ ϕ and sₜ ⊧ ψ at a common time t. If overlap=False, when ϕ stops being true, ψ starts being true. That is sₜ ⊧ ϕ and sₜ+₁ ⊧ ψ, but sₜ ¬⊧ ψ
    """

    def __init__(self, subformula1, subformula2, interval=None, overlap=True):
        super().__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        self.interval = interval
        if overlap == False:
            self.subformula2 = Eventually(subformula=subformula2, interval=[0,1])
        self.LARGE_NUMBER = 1E6

    def robustness_trace(self, signal, **kwargs):
        """
        Computing robustness trace of subformula1 U subformula2  (see paper)

        Args:
            signal: input signal for the formula. If using Expressions to define the formula, then inputs a tuple of signals corresponding to each subformula. If using Predicates to define the formula, then inputs is just a single torch.array. Not need for different signals for each subformula. Expected signal is size [batch_size, time_dim, x_dim]
            time_dim: axis for time_dim. Default: 1
            kwargs: Other arguments including time_dim, approx_method, temperature

        Returns:
            robustness_trace: torch.array. Same size as signal.
        """

        time_dim = 0
        LARGE_NUMBER = self.LARGE_NUMBER

        if isinstance(signal, tuple):
            # for formula defined using Expression
            assert signal[0].shape[time_dim] == signal[1].shape[time_dim]
            trace1 = self.subformula1(signal[0], **kwargs)
            trace2 = self.subformula2(signal[1], **kwargs)
            n_time_steps = signal[0].shape[time_dim]
        else:
            # for formula defined using Predicate
            trace1 = self.subformula1(signal, **kwargs)
            trace2 = self.subformula2(signal, **kwargs)
            n_time_steps = signal.shape[time_dim]

        Alw = AlwaysRecurrent(subformula=Identity(str(self.subformula1)))

        LHS = torch.tile(trace2, [n_time_steps, *([1]*len(trace2.shape))])
        # LHS = torch.permute_dims(torch.repeat(torch.expand_dims(trace2, -1), n_time_steps, dim=-1), [1,0])  # [sub_signal, t_prime]
        RHS = torch.ones_like(LHS) * -LARGE_NUMBER  # [sub_signal, t_prime]

        # Case 1, interval = [0, inf]
        if self.interval == None:
            for i in range(n_time_steps):
                RHS[i:,i] = Alw(trace1[i:])
                # RHS = RHS.at[i:,i].set(Alw(trace1[i:]))

        # Case 2 and 4: self.interval is [a, b], a ≥ 0, b < ∞
        elif self.interval[1] < torch.inf:
            a = self.interval[0]
            b = self.interval[1]
            for i in range(n_time_steps):
                end = i+b+1
                # RHS = RHS.at[i+a:end,i].set(Alw(trace1[i:end])[a:])
                RHS[i+a:end,i] = Alw(trace1[i:end])[a:]

        # Case 3: self.interval is [a, np.inf), a ≂̸ 0
        else:
            a = self.interval[0]
            for i in range(n_time_steps):
                # RHS = RHS.at[i+a:,i].set(Alw(trace1[i:])[a:])
                RHS[i+a:,i] = Alw(trace1[i:])[a:]

        return maxish(minish(torch.stack([LHS, RHS], dim=-1), dim=-1, keepdim=False, **kwargs), dim=-1, keepdim=False, **kwargs)


    def _next_function(self):
        """ next function is the input subformulas. For visualization purposes """
        return [self.subformula1, self.subformula2]

    def __str__(self):
        return  "(" + str(self.subformula1) + ")" + " U " + "(" + str(self.subformula2) + ")"




class DifferentiableAlways(STLFormula):
    def __init__(self, subformula, interval=None):
        super().__init__()

        self.interval = interval
        self.subformula = subformula
        self._interval = [0, torch.inf] if self.interval is None else self.interval

    def robustness_trace(self, signal, t_start, t_end, scale=1.0, padding="last", large_number=1E6, delta=1E-3, **kwargs):
        device = signal.device
        time_dim = 0  # assuming signal is [time_dim,...]
        signal = self.subformula(signal, padding=padding, large_number=large_number)
        T = signal.shape[time_dim]
        mask_value = large_number
        if self.interval is None:
            interval = [0,T-1]
        else:
            interval = self.interval
        signal_matrix = signal.reshape([T,1]) @ torch.ones([1,T], device=device)
        if padding == "last":
            pad_value = signal[-1]
        elif padding == "mean":
            pad_value = signal.mean(time_dim)
        else:
            pad_value = padding
        signal_pad = torch.ones([interval[1]+1, T], device=device) * pad_value
        signal_padded = torch.cat([signal_matrix, signal_pad], dim=time_dim)
        smooth_time_mask = smooth_mask(T, t_start, t_end, scale)# * (1 - delta) + delta
        padded_smooth_time_mask = torch.zeros([2 * T, T], device=device)
        for t in range(T):
            padded_smooth_time_mask[t:t+T,t] = smooth_time_mask
        masked_signal_matrix = torch.where(padded_smooth_time_mask > delta, signal_padded * padded_smooth_time_mask, mask_value)
        return minish(masked_signal_matrix, dim=time_dim, keepdim=False)

    def _next_function(self):
        """ next function is the input subformula. For visualization purposes """
        return [self.subformula]

    def __str__(self):
        return "◻ [a,b] ( " + str(self.subformula) + " )"

class DifferentiableEventually(STLFormula):
    def __init__(self, subformula, interval=None):
        super().__init__()

        self.interval = interval
        self.subformula = subformula
        self._interval = [0, torch.inf] if self.interval is None else self.interval

    def robustness_trace(self, signal, t_start, t_end, scale=1.0, padding="last", large_number=1E6, **kwargs):
        device = signal.device
        time_dim = 0  # assuming signal is [time_dim,...]
        delta = 1E-3
        signal = self.subformula(signal, padding=padding, large_number=large_number)
        T = signal.shape[time_dim]
        mask_value = -large_number
        if self.interval is None:
            interval = [0,T-1]
        else:
            interval = self.interval
        signal_matrix = signal.reshape([T,1]) @ torch.ones([1,T], device=device)
        if padding == "last":
            pad_value = signal[-1]
        elif padding == "mean":
            pad_value = signal.mean(time_dim)
        else:
            pad_value = padding
        signal_pad = torch.ones([interval[1]+1, T], device=device) * pad_value
        signal_padded = torch.cat([signal_matrix, signal_pad], dim=time_dim)
        smooth_time_mask = smooth_mask(T, t_start, t_end, scale)# * (1 - delta) + delta
        padded_smooth_time_mask = torch.zeros([2 * T, T], device=device)
        for t in range(T):
            padded_smooth_time_mask[t:t+T,t] = smooth_time_mask
        masked_signal_matrix = torch.where(padded_smooth_time_mask > delta, signal_padded * padded_smooth_time_mask, mask_value)
        return maxish(masked_signal_matrix, dim=time_dim, keepdim=False)

    def _next_function(self):
        """ next function is the input subformula. For visualization purposes """
        return [self.subformula]

    def __str__(self):
        return "♢ [a,b] ( " + str(self.subformula) + " )"