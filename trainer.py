class Trainer:
    def __init__(self, model, criterion, optimizer, device, tokenizer, train_loader, valid_loader, key):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        # Add tensorboard writer to Trainer class
        # self.writer = SummaryWriter()
        # tensorboard_log_dir = f'runs/lr={args.learning_rate}_without_pos_2700Imagespruned_v1_totalepochs={args.epochs}_perceptual_loss_factor={args.perceptual_loss_factor}_discfactor={args.disc_start}_codebook={args.num_codebook_vectors}_latentdim={args.latent_dim}_{time.strftime("%Y%m%d-%H%M%S")}'
        # writer = SummaryWriter(log_dir=tensorboard_log_dir)
        
        
        self.writer = SummaryWriter(log_dir=f'runsTransformer_1_lakhDataPoints/{key}')


    def train(self, epochs):
        
        def get_total_params(module: torch.nn.Module):
            total_params = 0
            for param in module.parameters():
                total_params += param.numel()
            return total_params

        print('Total parameters in model: {:,}'.format(get_total_params(self.model)))
        
        train_losses = np.zeros(epochs)
        validation_losses = np.zeros(epochs)
        train_perplexity_list = np.zeros(epochs)
        validation_perplexity_list = np.zeros(epochs)
        # print("print_number_of_trainable_model_parameters",
        #       self.print_number_of_trainable_model_parameters()) 
        for it in range(epochs):
            self.model.train()
            t0 = datetime.now()
            train_loss = [] 
            train_loss_scalar, train_correct_scalar, total_train_samples_scalar = 0, 0, 0
            for batch in self.train_loader:
                batch = {k:v.to(self.device) for k,v in batch.items()}
                self.optimizer.zero_grad()
                enc_input = batch['input_ids']
                enc_mask = batch['attention_mask']
                targets = batch['labels']
                dec_input, dec_mask = self.prepare_decoder_inputs(targets)
                outputs = self.model(enc_input, dec_input, enc_mask, dec_mask)
                loss = self.criterion(outputs.transpose(2,1), targets)
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())
                
                train_loss_scalar +=loss.item()
                _,predicted = torch.max(outputs, dim = 2)
                train_correct_scalar += (predicted == targets).sum().item()
                total_train_samples_scalar += targets.ne(self.tokenizer.pad_token_id).sum().item()
                
                
            train_loss = np.mean(train_loss)
            # test_loss = self.evaluate()
            avg_train_loss_scalar = train_loss_scalar / len(self.train_loader)
            train_accuracy_scalar = train_correct_scalar / total_train_samples_scalar
            train_perplexity_scalar = np.exp(avg_train_loss_scalar)
            
            
            avg_valid_loss_scalar, valid_accuracy_scalar, valid_perplexity_scalar,test_loss = self.evaluate()
            
            dt = datetime.now() - t0
            dt_seconds = dt.total_seconds()  # convert duration to seconds
            
             # Log the metrics to tensorboard
            self.writer.add_scalar('Train Loss', avg_train_loss_scalar, it)
            self.writer.add_scalar('Train Accuracy', train_accuracy_scalar, it)
            self.writer.add_scalar('Train Perplexity', train_perplexity_scalar, it)
            self.writer.add_scalar('Validation Loss', avg_valid_loss_scalar, it)
            self.writer.add_scalar('Validation Accuracy', valid_accuracy_scalar, it)
            self.writer.add_scalar('Validation Perplexity', valid_perplexity_scalar, it)
            self.writer.add_scalar('Epoch Duration', dt_seconds, it)
            
            
            train_losses[it] = train_loss
            validation_losses[it] = test_loss
            
            train_perplexity_list[it], validation_perplexity_list[it] =train_perplexity_scalar, valid_perplexity_scalar
            
            
            
            # Log the duration of this epoch to tensorboard
            
            
            print(f'Epoch {it+1}/{epochs}, Train Loss: {avg_train_loss_scalar:.4f}, \
                  Train Accuracy: {train_accuracy_scalar:.4f}, \
                  Train Perplexity: {train_perplexity_scalar:.4f}, \
                  Validation Loss: {avg_valid_loss_scalar:.4f}, \
                  Validation Accuracy: {valid_accuracy_scalar:.4f}, \
                  Validation Perplexity: {valid_perplexity_scalar:.4f}')
            
            # print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss: .4f}, Test Loss: {test_loss: .4f}, Duration: {dt}')
            
            
            # Check if the current epoch is a multiple of 50
            if (it + 1) % 50 == 0:
                checkpoint_path = "checkpoint_nlp_latest_epoch128_4_4_0dropout.pth"
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Model checkpoint saved to {checkpoint_path} at epoch {it+1}")
                
                
        self.writer.close()  # Close the writer after logging all losses
        
        # Save the model at the end of training
        # checkpoint_path = "checkpoint_nlp_1000_epoch.pth"
        # torch.save(self.model.state_dict(), checkpoint_path)
        # print(f"Model saved to {checkpoint_path}")
        return train_losses, validation_losses, train_perplexity_list, validation_perplexity_list

    def prepare_decoder_inputs(self, targets):
        dec_input = targets.clone().detach()
        dec_input = torch.roll(dec_input, shifts = 1, dims = 1)
        dec_input[:,0] = 65001
        dec_input = dec_input.masked_fill(dec_input==-100, self.tokenizer.pad_token_id )
        dec_mask = torch.ones_like(dec_input)
        dec_mask = dec_mask.masked_fill(dec_input == self.tokenizer.pad_token_id,0)
        return dec_input, dec_mask

    def evaluate(self):
        self.model.eval()
        test_loss = []
        valid_loss, valid_correct, total_valid_samples = 0, 0, 0
        for batch in self.valid_loader:
            batch = {k:v.to(self.device) for k,v in batch.items()}
            enc_input = batch['input_ids']
            enc_mask = batch['attention_mask']
            targets = batch['labels']
            dec_input, dec_mask = self.prepare_decoder_inputs(targets)
            outputs = self.model(enc_input, dec_input, enc_mask, dec_mask)
            loss = self.criterion(outputs.transpose(2,1), targets)
            test_loss.append(loss.item())
            
            
            valid_loss += loss.item()
            _, predicted = torch.max(outputs, dim=2)
            valid_correct += (predicted == targets).sum().item()
            total_valid_samples += targets.ne(self.tokenizer.pad_token_id).sum().item()
        avg_valid_loss = valid_loss / len(self.valid_loader)
        valid_accuracy = valid_correct / total_valid_samples
        valid_perplexity = np.exp(avg_valid_loss)
        # return 
            
        return avg_valid_loss, valid_accuracy, valid_perplexity,np.mean(test_loss)