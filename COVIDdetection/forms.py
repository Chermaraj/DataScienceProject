from django import forms
from COVIDdetection.models import Document

#class UploadFileForm(forms.Form):
 #   docfile = forms.FileField(
  #      label='Select a file',
   #     help_text='max. 42 megabytes'
    #)'''

class UploadFileForm(forms.ModelForm):
    
    patient  = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control',
                                      'label': 'Enter patient name', 'size':'20',
                                      'maxlength': '20', 'pattern':'[A-Za-z ]+',
                                      'required': True}))

    docfile_beforeAnalysis = forms.ImageField(
                                     label='Select a file',
                                     help_text='max. 42 megabytes')

   
    
    class Meta:
        model = Document
        fields = ('patient', 'docfile_beforeAnalysis')