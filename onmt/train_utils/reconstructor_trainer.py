


class ReconstructorTrainer(ReconstructorTrainer):

    def __init__(self, model, loss_function, trainData, validData, dataset, opt):
        super().__init__(model, loss_function, trainData, validData, dataset, opt)
        self.optim = onmt.NoamOptim(opt)
        
        if self.cuda:
           torch.cuda.set_device(self.opt.gpus[0])
           torch.manual_seed(self.opt.seed)
           self.loss_function = self.loss_function.cuda()
           self.model = self.model.cuda()
        
        self.optim.set_parameters(self.model.parameters())

    def save(self, epoch, valid_ppl, batchOrder=None, iteration=-1):
        
        opt, dataset = self.opt, self.dataset
        model = self.model
        

        model_state_dict = self.model.state_dict()
        optim_state_dict = self.optim.state_dict()
                
        #  drop a checkpoint
        checkpoint = {
                'model': model_state_dict,
                'dicts': dataset['dicts'],
                'opt': opt,
                'epoch': epoch,
                'iteration' : iteration,
                'batchOrder' : batchOrder,
                'optim': optim_state_dict
        }
        
        file_name = '%s_ppl_%.2f_e%.2f.pt' % (opt.save_model, valid_ppl, epoch)
        print('Writing to %s' % file_name)
        torch.save(checkpoint, file_name)
        
    def eval(self, data):
        total_loss = 0
        total_words = 0
                
        batch_order = data.create_order(random=False)
        self.model.eval()
        """ New semantics of PyTorch: save space by not creating gradients """
        with torch.no_grad():
            for i in range(len(data)):
                    
                samples = data.next()
                
                batch = self.to_variable(samples[0])
                
                """ outputs can be either 
                        hidden states from decoder or
                        prob distribution from decoder generator
                """
                outputs = self.model(batch)
                targets = batch[1][1:]
                
                loss_data, grad_outputs = self.loss_function(outputs, targets, generator=self.model.generator, backward=False)
                
#~ 
                total_loss += loss_data
                total_words += targets.data.ne(onmt.Constants.PAD).sum()

        self.model.train()
        return total_loss / total_words
        
    def train_epoch(self, epoch, resume=False, batchOrder=None, iteration=None):
        
        opt = self.opt
        trainData = self.trainData
        
        # Clear the gradients of the model
        # self.runner.zero_grad()
        self.model.zero_grad()

        if opt.extra_shuffle and epoch > opt.curriculum:
            trainData.shuffle()

        # Shuffle mini batch order.
        
        if resume:
            trainData.batchOrder = batchOrder
            trainData.set_index(iteration)
            print("Resuming from iteration: %d" % iteration)
        else:
            batchOrder = trainData.create_order()
            iteration = 0
        # if batchOrder is not None:
            # batchOrder = trainData.create_order()
        # else:
            # trainData.batchOrder = batchOrder
            
        # if iteration is not None and iteration > -1:
            # trainData.set_index(iteration)
            # print("Resuming from iteration: %d" % iteration)

        total_loss, total_words = 0, 0
        report_loss, report_tgt_words = 0, 0
        report_src_words = 0
        start = time.time()
        nSamples = len(trainData)
        dataset = self.dataset
        
        counter = 0
        num_accumulated_words = 0
        
        for i in range(iteration, nSamples):

            curriculum = (epoch < opt.curriculum)
            
            samples = trainData.next(curriculum=curriculum)
                        
            batch = self.to_variable(samples[0])
            
            outputs = self.model(batch)
                
            targets = batch[1][1:]
            
            loss_data, grad_outputs = self.loss_function(outputs, targets, generator=self.model.generator, backward=True)
            
            outputs.backward(grad_outputs)
            
            src_size = batch[0].data.ne(onmt.Constants.PAD).sum()
            tgt_size = targets.data.ne(onmt.Constants.PAD).sum()
            
            counter = counter + 1 
            num_accumulated_words += tgt_size
            
            # We only update the parameters after getting gradients from n mini-batches
            # simulating the multi-gpu situation
            if counter == opt.virtual_gpu:
                # Update the parameters.
                grad_denom=num_accumulated_words if self.opt.normalize_gradient else 1
                self.optim.step(grad_denom=grad_denom)
                self.model.zero_grad()
                counter = 0
                num_accumulated_words = 0
                if opt.save_every > 0 and num_updates % opt.save_every == -1 % opt.save_every :
                    valid_loss = self.eval(self.validData)
                    valid_ppl = math.exp(min(valid_loss, 100))
                    print('Validation perplexity: %g' % valid_ppl)
                    
                    ep = float(epoch) - 1. + ((float(i) + 1.) / nSamples)
                    
                    self.save(ep, valid_ppl, batchOrder=batchOrder, iteration=i)
            

            num_words = tgt_size
            report_loss += loss_data
            report_tgt_words += num_words
            report_src_words += src_size
            total_loss += loss_data
            total_words += num_words
            
            optim = self.optim
            num_updates = optim._step
            
            
            if i == 0 or (i % opt.log_interval == -1 % opt.log_interval):
                print(("Epoch %2d, %5d/%5d; ; ppl: %6.2f ; lr: %.7f ; num updates: %7d " +
                       "%5.0f src tok/s; %5.0f tgt tok/s; %s elapsed") %
                      (epoch, i+1, len(trainData),
                       math.exp(report_loss / report_tgt_words),
                       optim.getLearningRate(),
                       optim._step,
                       report_src_words/(time.time()-start),
                       report_tgt_words/(time.time()-start),
                       str(datetime.timedelta(seconds=int(time.time() - self.start_time)))))

                report_loss, report_tgt_words = 0, 0
                report_src_words = 0
                start = time.time()
            
            
        return total_loss / total_words
    
    
    
    def run(self, save_file):
        
        assert save_file is not None, "Training reconstructor must start from a checkpoint"
        
        opt = self.opt
        model = self.model
        dataset = self.dataset
        optim = self.optim
        
        # Try to load the save_file
        checkpoint = torch.load(save_file)
        
        print('Loading model and optim from checkpoint at %s' % save_file)
        self.model.load_state_dict(checkpoint['model'])
        self.optim.load_state_dict(checkpoint['optim'])
        batchOrder = checkpoint['batchOrder']
        iteration = checkpoint['iteration'] + 1
        opt.start_epoch = int(math.floor(float(checkpoint['epoch'] + 1)))   
        del checkpoint['model']
        del checkpoint['optim']
        del checkpoint
        resume=True
        
        self.reconstructor = 
        #~ else:
            #~ batchOrder = None
            #~ iteration = None
            #~ print('Initializing model parameters')
            #~ init_model_parameters(model, opt)
            #~ resume=False
        
        
        valid_loss = self.eval(self.validData)
        valid_ppl = math.exp(min(valid_loss, 100))
        print('Validation perplexity: %g' % valid_ppl)
        
        self.start_time = time.time()
        
        for epoch in range(opt.start_epoch, opt.start_epoch + opt.epochs):
            print('')

            #  (1) train for one epoch on the training set
            train_loss = self.train_epoch(epoch, resume=resume,
                                                 batchOrder=batchOrder,
                                                 iteration=iteration)
            train_ppl = math.exp(min(train_loss, 100))
            print('Train perplexity: %g' % train_ppl)

            #  (2) evaluate on the validation set
            valid_loss = self.eval(self.validData)
            valid_ppl = math.exp(min(valid_loss, 100))
            print('Validation perplexity: %g' % valid_ppl)
            
            
            self.save(epoch, valid_ppl)
            batchOrder = None
            iteration = None
            resume = False
        
        
    
    
    
