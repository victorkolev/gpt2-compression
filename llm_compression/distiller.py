# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Knowledge Distillation Helper class
"""
# import abc

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2Model, GPT2Tokenizer

class TeacherWrapper(nn.Module):
    """Model distillation teacher wrapper class"""

    def __init__(self, teacher, *, ce_alpha=0., ce_temperature=1., convert_parameters=True, keep_gradients=False):
        super().__init__()
        self.teacher = teacher
        self.keep_gradients = keep_gradients
        self.ce_alpha = ce_alpha
        self.ce_temperature = ce_temperature
        if convert_parameters:
            self.convert_parameters_to_buffers()
        self._output = None

    def forward(self, *args, **kwargs):
        """Compute teacher forward and return teacher output"""
        self.teacher.eval()
        # In case output wasn't delted yet, delete output to make room for new output
        if hasattr(self, '_output'):
            del self._output
        with torch.set_grad_enabled(self.keep_gradients):
            self._output = self.teacher(*args, **kwargs)
        return self._output

    def convert_parameters_to_buffers(self):
        """Convert teacher module parameters to module buffers"""
        for m in self.teacher.modules():
            for n, p in list(m.named_parameters(recurse=False)):
                delattr(m, n)
                m.register_buffer(n, p.data)

    def compute_cross_entropy_loss(self, student_outputs, teacher_outputs):
        """Compute cross entropy loss"""
        return F.kl_div(
            input=F.log_softmax(student_outputs / self.ce_temperature, dim=-1),
            target=F.softmax(teacher_outputs / self.ce_temperature, dim=-1),
            reduction="batchmean"
        ) * (self.ce_temperature ** 2)

    def compute_distill_loss_callback(self, student_outputs, teacher_outputs=None):
        """Compute the distillation loss w.r.t teacher"""
        return self.compute_cross_entropy_loss(student_outputs, teacher_outputs) * self.ce_alpha

    def compute_distill_loss(self, student_outputs, teacher_outputs=None):
        """Compute the distillation loss w.r.t teacher scaled with teacher alpha"""
        teacher_outputs = self.get_teachers_outputs(teacher_outputs)
        distill_loss = self.compute_distill_loss_callback(
            student_outputs, teacher_outputs)
        # After calculation of the distillation loss we delete the output to conserve memory
        del self._output
        return distill_loss

    def get_teachers_outputs(self, teacher_outputs=None):
        """Get teacher's cached outputs"""
        if teacher_outputs is None:
            teacher_outputs = self._output
        return teacher_outputs


class DistillationModelWrapper(nn.Module):
    """
    Model distillation wrapper combining student and teachers to a single model that 
    outputs the knowledge distillation loss in the forward pass
    """

    def __init__(self, student, teachers, *, alpha_student=1., **_):
        super().__init__()
        self.student = student
        teachers = teachers if isinstance(teachers, list) else [teachers]
        for teacher in teachers:
            if not isinstance(teacher, TeacherWrapper):
                raise RuntimeError(
                    "Recieved a teacher not wrapped with TeacherWrapper class")
        self.teachers = nn.ModuleList(teachers)
        self.alpha_student = alpha_student

    def forward(self, *args, **kwargs):
        if self.training:
            for teacher in self.teachers:
                teacher(*args, **kwargs)
        return self.student(*args, **kwargs)

    def compute_loss(self, student_loss, student_outputs):
        """Compute combined loss of student with teachers"""
        loss = student_loss
        if self.training:
            loss *= self.alpha_student
            for teacher in self.teachers:
                loss += teacher.compute_distill_loss(student_outputs)
        return loss


def playground_main():
    teacher_play_model = nn.Sequential(
        nn.Linear(10, 2)
    )
    student_play_model = nn.Sequential(
        nn.Linear(10, 2)
    )

    teacher = TeacherWrapper(teacher_play_model, ce_alpha=0.5, ce_temperature=2.0)
    distillation_model = DistillationModelWrapper(student_play_model, teacher, alpha_student=0.5)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(student_play_model.parameters())
    m = 128
    for e in range(1000):
        inputs, labels = torch.rand(m, 10), torch.rand(m, 2)
        distillation_model.train()
        # Calculate student loss w.r.t labels as you usually do
        student_outputs = distillation_model(inputs)
        loss_wrt_labels = criterion(student_outputs, labels)
        # Add knowledge distillation term
        loss = distillation_model.compute_loss(loss_wrt_labels, student_outputs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if e % 100 == 0:
            print(loss.item())

def get_data():
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    return dataset

def gpt2_playground():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model = GPT2Model.from_pretrained("distilgpt2").to(device)
    text = "hello world how are you"
    encoded_input = tokenizer(text, return_tensors="pt").to(device)
    out = model(**encoded_input)
    words_reps = out[0]
    print(words_reps.shape)


def gpt2_main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model_teacher = GPT2Model.from_pretrained("distilgpt2").to(device)
    model = GPT2Model.from_pretrained("distilgpt2").to(device)

    teacher = TeacherWrapper(model_teacher, ce_alpha=0.5, ce_temperature=2.0)
    distillation_model = DistillationModelWrapper(model, teacher, alpha_student=0.5)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    m = 128
    data = get_data()
    print(type(data["train"]))
    print(data["train"][1:10])
    examples = [x["text"] for x in data["train"] if len(x["text"]) > 0]
    print("EXAMPLES", examples[:10])
    for e in range(4):
        for i in range(0, len(examples)):
            # input = examples[i:i+m]
            # # tokenizer(input, return_tensors="pt").to(device)
            # for x in examples[i:i+m]:
            #     print(x)
            # encoded_input = [tokenizer(x, return_tesnor="pt") for x in examples[i:i+m]]
            # print("encoded input shape", len(encoded_input))
            # encoded_input = encoded_input.to(device)
        # text = "hello world how are you"
        #     encoded_input = tokenizer(text, return_tensors="pt").to(device)
            text = examples[i]
            encoded_input = tokenizer(text, return_tensors="pt").to(device)
            teacher_out = teacher(**encoded_input)
            teacher_words_reps = teacher_out[0]
            teacher_labels = teacher_out[1]

            distillation_model.train()
            student_out = distillation_model(**encoded_input)
            student_word_reps = student_out[0]
            student_labels = student_out[1]

            # Calculate student loss w.r.t labels as you usually do
            loss_wrt_labels = criterion(student_word_reps, teacher_words_reps)
            # Add knowledge distillation term
            loss = distillation_model.compute_loss(loss_wrt_labels, student_word_reps)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # if e % 100 == 0:
            print(loss.item())

if __name__ == "__main__":
    # get_data()
    # playground_main()
    gpt2_main()
