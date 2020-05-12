
# conv - adding 

class CNN_D(torch.nn.Module):

    def __init__(self, ninput, noutput, filter_sizes='1-2-3-4-5', n_filters=128, activation=nn.LeakyReLU(0.2), gpu=False):
        super(CNN_D, self).__init__()

        self.n_filters = n_filters
        self.ninput = ninput
        self.noutput = noutput
        #FIX THIS
        self.filter_sizes =  [int(x) for x in filter_sizes.split('-')]
#         print(self.filter_sizes)
        print(ninput, noutput)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=self.n_filters, padding=int((filter_size - 1)/2), kernel_size=filter_size) for filter_size in self.filter_sizes])
        self.activation=nn.LeakyReLU(.2)
        self.dropout = nn.Dropout(0.3)
        # flatten before feeding into fully connected
        self.fc = nn.Linear(self.ninput*noutput*self.n_filters*len(self.filter_sizes), 1, bias=True)  # calculate size here,1 bias = True)

   
    def forward(self, x):
        # x shape is (N, l_in)
#         print("x shape is: {}".format(x.shape))
#         print(x)
#        c_in = 1
        # x shape is (N, C_in, l_in) = (64, 1, 128)
        x = x.unsqueeze(1)
#         print("x reshaped is: {}".format(x.shape))
        # x is shape (N, C_in, 1)
        outputs = []
        for i in range(len(self.convs)):
            h = self.activation(self.convs[i](x))
#             print("h shape is: {}".format(h.shape))
            # output shape is 
#             h = h.squeeze()
#             h shape should be (N, out_channels, size)
#             print(h.shape)
#             pooled, pooled_idx = torch.max(h, dim=2)
            #             pooled = pooled.contiguous().view(-1, self.n_filters).contiguous()
#             print("pool shape is: {}".format(pooled.shape))
            outputs.append(h)
        outputs=torch.cat(outputs, 1)
#         print("outputs shape is: {}".format(outputs.shape))

#         print("outputs shape is: {}".format(outputs.shape))

#         outputs=self.dropout(x).view(-1).contiguous()
        outputs=outputs.view(-1).contiguous()
#         print("outputs reshaped is: {}".format(outputs.shape))

#         print("reshaped outputs shape is: {}".format(outputs.shape))
#         outputs=outputs.view(-1).contiguous()
#         print(outputs)
#         print(outputs.shape)
#         print("fc shape is: {}".format(self.ninput*self.noutput*self.noutput*len(self.filter_sizes)))
        logits=self.fc(outputs)
#         print("logits shape is: {}".format(logits.shape))

#         print("logits shape is: {}".format(logits.shape))
#         print(torch.sum(logits).contiguous())

        return logits






#Version 2 - using crosss entropy loss

class CNN_D(torch.nn.Module):

    def __init__(self, ninput, noutput, filter_sizes='1-2-3-4-5', n_filters=128, activation=nn.LeakyReLU(0.2), gpu=False):
        super(CNN_D, self).__init__()

        self.n_filters = n_filters
        self.ninput = ninput
        self.noutput = noutput
        #FIX THIS
        self.filter_sizes =  [int(x) for x in filter_sizes.split('-')]
#         print(self.filter_sizes)
        print(ninput, noutput)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=self.n_filters, padding=int((filter_size - 1)/2), kernel_size=filter_size) for filter_size in self.filter_sizes])
        self.activation=nn.LeakyReLU(.2)
        self.dropout = nn.Dropout(0.3)
        # flatten before feeding into fully connected
        self.fc = nn.Linear(self.ninput*noutput*self.n_filters*len(self.filter_sizes), 1, bias=True)  # calculate size here,1 bias = True)
        self.ce = nn.CrossEntropyLoss()


   
    def forward(self, x):
        # x shape is (N, l_in)
#         print("x shape is: {}".format(x.shape))
#         print(x)
#        c_in = 1
        # x shape is (N, C_in, l_in) = (64, 1, 128)
        x = x.unsqueeze(1)
#         print("x reshaped is: {}".format(x.shape))
        # x is shape (N, C_in, 1)
        outputs = []
        for i in range(len(self.convs)):
            h = self.activation(self.convs[i](x))
#             print("h shape is: {}".format(h.shape))
            # output shape is 
#             h = h.squeeze()
#             h shape should be (N, out_channels, size)
#             print(h.shape)
#             pooled, pooled_idx = torch.max(h, dim=2)
            #             pooled = pooled.contiguous().view(-1, self.n_filters).contiguous()
#             print("pool shape is: {}".format(pooled.shape))
            outputs.append(h)
        outputs=torch.cat(outputs, 1)
#         print("outputs shape is: {}".format(outputs.shape))

#         print("outputs shape is: {}".format(outputs.shape))

#         outputs=self.dropout(x).view(-1).contiguous()
        outputs=outputs.view(-1).contiguous()
#         print("outputs reshaped is: {}".format(outputs.shape))

#         print("reshaped outputs shape is: {}".format(outputs.shape))
#         outputs=outputs.view(-1).contiguous()
#         print(outputs)
#         print(outputs.shape)

#         print("fc shape is: {}".format(self.ninput*self.noutput*self.noutput*len(self.filter_sizes)))
        
        logits = self.ce(outputs, torch.zeros(outputs.shape))
#         logits=self.fc(outputs)
#         print("logits shape is: {}".format(logits.shape))

#         print("logits shape is: {}".format(logits.shape))
#         print(torch.sum(logits).contiguous())

        return logits



# version 3 

class CNN_D(torch.nn.Module):

    def __init__(self, ninput, noutput, filter_sizes='1-2-3-4-5', n_filters=128, activation=nn.LeakyReLU(0.2), gpu=False):
        super(CNN_D, self).__init__()

        self.n_filters = n_filters
        self.ninput = ninput
        self.noutput = noutput
        #FIX THIS
        self.filter_sizes =  [int(x) for x in filter_sizes.split('-')]
#         print(self.filter_sizes)
        print(ninput, noutput)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=self.n_filters, padding=int((filter_size - 1)/2), kernel_size=filter_size) for filter_size in self.filter_sizes])
        self.activation=nn.LeakyReLU(.2)
        self.dropout = nn.Dropout(0.3)
        # flatten before feeding into fully connected
        self.fc = nn.Linear(self.ninput*noutput*self.n_filters*len(self.filter_sizes), 1, bias=True)  # calculate size here,1 bias = True)
        self.m = nn.Sigmoid()
        self.ce = nn.bce()


   
    def forward(self, x):
        # x shape is (N, l_in)
#         print("x shape is: {}".format(x.shape))
#         print(x)
#        c_in = 1
        # x shape is (N, C_in, l_in) = (64, 1, 128)
        x = x.unsqueeze(1)
#         print("x reshaped is: {}".format(x.shape))
        # x is shape (N, C_in, 1)
        outputs = []
        for i in range(len(self.convs)):
            h = self.activation(self.convs[i](x))
#             print("h shape is: {}".format(h.shape))
            # output shape is 
#             h = h.squeeze()
#             h shape should be (N, out_channels, size)
#             print(h.shape)
#             pooled, pooled_idx = torch.max(h, dim=2)
            #             pooled = pooled.contiguous().view(-1, self.n_filters).contiguous()
#             print("pool shape is: {}".format(pooled.shape))
            outputs.append(h)
        outputs=torch.cat(outputs, 1)
#         print("outputs shape is: {}".format(outputs.shape))

#         print("outputs shape is: {}".format(outputs.shape))

#         outputs=self.dropout(x).view(-1).contiguous()
        outputs=outputs.view(-1).contiguous()
#         print("outputs reshaped is: {}".format(outputs.shape))

#         print("reshaped outputs shape is: {}".format(outputs.shape))
#         outputs=outputs.view(-1).contiguous()
#         print(outputs)
#         print(outputs.shape)

#         print("fc shape is: {}".format(self.ninput*self.noutput*self.noutput*len(self.filter_sizes)))
        
        logits = self.ce(self.m(outputs), torch.zeros(outputs.shape))
#         print("logits shape is: {}".format(logits.shape))
        return logits

