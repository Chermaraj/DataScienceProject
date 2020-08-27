from django.db import models
import os
#from gdstorage.storage import GoogleDriveStorage, GoogleDrivePermissionType, GoogleDrivePermissionRole, GoogleDriveFilePermission

# Define Google Drive Storage
#gd_storage = GoogleDriveStorage()

# Create your models here.
def get_upload_path(instance, filename):
    return os.path.join(
      "patient/BeforeAnalysis/%s" % instance.patient, filename)

def get_temp_path(instance, filename):
    return os.path.join(
      "media/patient/BeforeAnalysis/%s" % instance.patient, filename)

def get_result_path(instance, filename):
    return os.path.join(
      "patient/AfterAnalysis/%s" % instance.patient, filename)



class Document(models.Model):
    doc_id = models.AutoField( primary_key=True)
    patient = models.CharField(unique=True, max_length=20) 
    #docfile_beforeAnalysis = models.ImageField(upload_to=lambda instance, filename: 'patient/BeforeAnalysis/{0}/{1}'.format(instance.patient, filename))
    docfile_beforeAnalysis = models.ImageField(upload_to=get_upload_path)
    docfile_forModel = models.ImageField(upload_to=get_temp_path)
    docfile_afterAnalysis = models.CharField(max_length=200)
    result =  models.CharField(max_length=30) 
       
    def filename(self):
        return os.path.basename(self.file.name)