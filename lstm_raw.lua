require 'dp'
require 'torch'
require 'rnn'

hiddenSize=100
inputSize=2
outputSize=10

dropoutProb=0.5

lstm=nn.LSTM(inputSize,hiddenSize)
model=nn.Sequential()
model:add(nn.SplitTable(1,2))
model:add(nn.Sequencer(lstm))
model:add(nn.Sequencer(nn.Dropout(dropoutProb)))
model:add(nn.SelectTable(-1))
model:add(nn.Linear(hiddenSize,outputSize))
model:add(nn.LogSoftMax())

criterion=nn.ClassNLLCriterion()

input=torch.Tensor( {{0,0},{1,1},{2,2},{3,3}}  )
output_actual=torch.Tensor({1,0,0,0,0,0,0,0,0,0})
output=model:forward(input)
learning_rate=2e-3
learning_rate_decay_after=10
decay=0.97
lr_decay=0.95


max_epochs=100
ntrain=

parameters_lstm,gradParameters_lstm=lstm:getParameters()

--print(parameters_lstm)
--print("--------------------------------------\n")
--print(gradParameters_lstm)

function feval_lstm(x)
	collectgarbage()
	if x~=parameters_lstm then
		parameters_lstm:copy(x)
	end

	gradParameters_lstm:zero()

	inputs,targets=loader  -- Get the next minibatch of data
	
	-- forward pass
	local outputs=lstm:forward(inputs)
	local f=criterion:forward( outputs,targets )

	-- backward pass
	local df_do=criterion:backward( outputs,targets )

	lstm:backward( inputs, df_do )
	return f,gradParameters_lstm
end

train_losses={}
val_losses={}

local optim_state={ learningRate=learning_rate, alpha=lr_decay }

local iterations=max_epochs*ntrain

for i=1, iterations do
	local epoch=i/ntrain
	local _,loss=optim.rmsprop(feval_lstm,parameters_lstm,optim_state)
	local train_loss=loss[1]
	train_losses[i]=train_loss

	-- exponential learning rate decay
	if i % ntrain==0 and epoch>=learning_rate_decay_after then
		local decay_factor=learning_rate_decay
		optim_state.learningRate=optim_state.learningRate*decay_factor
	end
	
	if i%eval_val_every==0 or i==iterations then 
		local savefile=string.format('%s/model_iter%d.t7',i)
		print('saving checkpoint to '.. savefile)
		local checkpoint={}
		checkpoint.lstm=lstm
		torch.save(savefile,checkpoint)
	end	

end

function loader(x)



end

