
import os

from django.shortcuts import HttpResponse, render, redirect
from django.core.files.storage import FileSystemStorage
from COVIDdetection.models import  Document
from COVIDdetection.learningmodel import learning_model
#from COVIDdetection.models import Map
from COVIDdetection.forms import UploadFileForm

def index(request):
    form = UploadFileForm()
    return render(request,'COVIDdetection/Base.html', {'form': form})


def uploadImage(request):

    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        print(form.is_valid())
        if form.is_valid():
            newdoc = Document(docfile_beforeAnalysis = request.FILES['docfile_beforeAnalysis'])
            newdoc.docfile_forModel = request.FILES['docfile_beforeAnalysis']
            newdoc.patient = form.cleaned_data.get('patient')
            patient_name = form.cleaned_data.get('patient')
            newdoc.save() 
            filePath =  newdoc.docfile_forModel
            print(filePath)
            modelResult,file_path = learning_model(filePath,patient_name)
            updatedoc = Document.objects.get(patient=patient_name)
            print(modelResult)
            print(file_path)
            newPath = file_path.replace('\\', '/')
            newPath = '/'+newPath
            updatedoc.result = modelResult
            updatedoc.docfile_afterAnalysis= newPath
            updatedoc.save()

            return redirect('patientList', patient_name)
            #return render(request, 'COVIDdetection/patientList.html', {'patient': patient_name})
            #return render(request,'COVIDdetection/Base.html', {'form': form})

        else:
            form = UploadFileForm()
    else:
        form = UploadFileForm()

    context = {'form': form}
    return render(request,'COVIDdetection/Base.html', context)

def patientList(request,patient_name):

   #if request.method == 'GET':
     #patient_name = get_object_or_404('patient_name')
     #patient_name = request.GET['patient']
     print('here' +patient_name)

     if patient_name is None:

        documents = Document.objects.all()
     else:
        documents = Document.objects.filter(patient=patient_name)

     context = {'documents': documents}

     return render(request,'COVIDdetection/PatientList.html', context)

def team(request):
    return render(request,'COVIDdetection/Team.html')


'''def uploadImage(request):
    if request.method == 'POST':
        uploadedFile = request.FILES['myfile']
        fs = FileSystemStorage()
        fs.save(uploadedFile.name,uploadedFile)
    return render(request,'COVIDdetection/Base.html')'''


