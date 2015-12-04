require 'torch'
require 'nn'
require 'nngraph'


function get_rnn(input_size, rnn_size)
  
    -- there are n+1 inputs (hiddens on each layer and x)
    local input = nn.Identity()()
    local prev_h = nn.Identity()()

    -- RNN tick
    local i2h = nn.Linear(input_size, rnn_size)(input)
    local h2h = nn.Linear(rnn_size, rnn_size)(prev_h)
    local added_h = nn.CAddTable()({i2h, h2h})
    local next_h = nn.Tanh()(added_h)
    
    nngraph.annotateNodes()
    return nn.gModule({input, prev_h}, {next_h})
end

function get_dkt_rnn(n_hidden, n_input, n_questions)
  -- The transfer parameters
	local transfer = nn.Linear(n_hidden, n_hidden)

	-- The first layer 
	local start = nn.Linear(1, n_hidden)

	-- Prototypical layer
	local inputM = nn.Identity()(); -- the memory input
	local inputX = nn.Identity()(); -- the last student activity
	local inputY = nn.Identity()(); -- the next question answered
	local truth  = nn.Identity()(); -- whether the next question is correct

	local linM   = transfer:clone('weight', 'bias')(inputM);
	local linX   = nn.Linear(n_input, n_hidden)(inputX);
	local madd   = nn.CAddTable()({linM, linX});
	local hidden = nn.Tanh()(madd);
	
	local predInput = nil
  predInput = nn.Dropout()(hidden)

	local linY = nn.Linear(n_hidden, n_questions)(predInput);
	local pred_output   = nn.Sigmoid()(linY);
	local pred          = nn.Sum(2)(nn.CMulTable()({pred_output, inputY}));
	local err           = nn.BCECriterion()({pred, truth})

	linX:annotate{name='linX'}
	linY:annotate{name='linY'}
	linM:annotate{name='linM'}
  nngraph.annotateNodes()
	local layer = nn.gModule({inputM, inputX, inputY, truth}, {pred, err, hidden});

	return layer
end

function test1()
  net = nn.Sequential()
  net:add(nn.SpatialConvolution(1, 6, 5, 5)) -- 1 input image channel, 6 output channels, 5x5 convolution kernel
  net:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
  net:add(nn.SpatialConvolution(6, 16, 5, 5))
  net:add(nn.SpatialMaxPooling(2,2,2,2))
  net:add(nn.View(16*5*5))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
  net:add(nn.Linear(16*5*5, 120))             -- fully connected layer (matrix multiplication between input and weights)
  net:add(nn.Linear(120, 84))
  net:add(nn.Linear(84, 10))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
  net:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classification problems

  --print('Lenet5\n' .. net:__tostring());
  input = torch.rand(1,32,32) -- pass a random tensor as input to the network
  output = net:forward(input)
  print(output)
  
  net:zeroGradParameters() -- zero the internal gradient buffers of the network (will come to this later)
  gradInput = net:backward(input, torch.rand(10))
  print(#gradInput)
end


test1()