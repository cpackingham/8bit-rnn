import torch
from torch.autograd import Variable
import os
from model import Generator
from librosa.output import write_wav

# Based on torch.utils.trainer.Trainer code.
# Allows multiple inputs to the model, not all need to be Tensors.
class Trainer(object):

    last_pattern = 'ep{}-it{}'
    best_pattern = 'best-ep{}-it{}'
    pattern = 'ep{}-s{}.wav'

    def __init__(self, model, criterion, optimizer, dataset, checkpoints_path, samples_path, n_samples, sample_length, sample_rate,test_dataset, val_dataset, cuda=False):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = dataset
        self.checkpoints_path = checkpoints_path
        self.samples_path = samples_path
        self.n_samples = n_samples
        self.sample_length = sample_length
        self.sample_rate = sample_rate
        self.cuda = cuda
        self.iterations = 0
        self.stats = {"best_loss": float('inf')}
        self.epochs = 0
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
        self._best_val_loss = float('inf')
        self.generate = Generator(self.model.model, self.cuda)

    def run(self, epochs=1):
        for self.epochs in range(self.epochs + 1, self.epochs + epochs + 1):
            self.train()
            self.epochs = self.epochs + 1
          #  self.model.eval()

           # val_stats['last'] = self._evaluate(self.val_dataset)
           # self.stats['validation_loss'] = val_stats
           # test_stats['last'] = self._evaluate(self.test_dataset)
           # self.stats['test_loss'] = test_stats

    def train(self):
        for (self.iterations, data) in \
                enumerate(self.dataset, self.iterations + 1):
            batch_inputs = data[: -1]
            batch_target = data[-1]

            def wrap(input):
                if torch.is_tensor(input):
                    input = Variable(input)
                    if self.cuda:
                        input = input.cuda()
                return input
            batch_inputs = list(map(wrap, batch_inputs))

            batch_target = Variable(batch_target)
            if self.cuda:
                batch_target = batch_target.cuda()

            def closure():
                #========PROCESS BATCH======
                batch_output = self.model(*batch_inputs)
                loss = self.criterion(batch_output, batch_target)
                loss.backward()
                self.cur_val_loss = loss.data[0]
                print("loss: " + str(loss))  
                return loss
            print("epoch number: " + str(self.epochs))
            print("finished batch")
            self.optimizer.zero_grad()
            self.optimizer.step(closure)
           # samples = self.generate(self.n_samples, self.sample_length) \
            #  .cpu().float().numpy()
            #for i in range(self.n_samples):
            #  write_wav(
            #      os.path.join(
            #          self.samples_path, self.pattern.format(self.epochs, self.iterations)
            #      ),
            #      samples[i, :], sr=self.sample_rate, norm=True
            #  )
            #========FINISHED PROCESSING BATCH==========
            torch.save(
              self.model.state_dict(),
              os.path.join(
                self.checkpoints_path,
                self.last_pattern.format(self.epochs, self.iterations)
              )
            )

            if cur_val_loss < self._best_val_loss:
                self._clear(self.best_pattern.format('*', '*'))
                torch.save(
                    self.trainer.model.state_dict(),
                    os.path.join(
                        self.checkpoints_path,
                        self.best_pattern.format(
                           self.epochs, self.iterations
                        )
                    )
                )
                self._best_val_loss = cur_val_loss

    def _clear(self, pattern):
        pattern = os.path.join(self.checkpoints_path, pattern)
        for file_name in glob(pattern):
            os.remove(file_name)
            print("saved")

    def _evaluate(self, dataset):
        loss_sum = 0
        n_examples = 0
        for data in dataset:
            batch_inputs = data[: -1]
            batch_target = data[-1]
            batch_size = batch_target.size()[0]

            def wrap(input):
                if torch.is_tensor(input):
                    input = Variable(input, volatile=True)
                    if self.cuda:
                        input = input.cuda()
                return input
            batch_inputs = list(map(wrap, batch_inputs))

            batch_target = Variable(batch_target, volatile=True)
            if self.trainer.cuda:
                batch_target = batch_target.cuda()

            batch_output = self.model(*batch_inputs)
            loss_sum += self.criterion(batch_output, batch_target) \
                                    .data[0] * batch_size

            n_examples += batch_size

        return loss_sum / n_examples
