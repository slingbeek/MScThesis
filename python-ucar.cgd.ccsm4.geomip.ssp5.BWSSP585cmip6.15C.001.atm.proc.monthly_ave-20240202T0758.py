#!/usr/bin/env python3

import os
import sys
import shutil
import hashlib
import urllib.request
import time
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from platform import python_version

################################################################
#
# Generated by: NCAR Climate Data Gateway
# Created: 2024-02-02T07:58:44-07:00
#
# Your download selection includes data that might be secured using API Token based
# authentication. Therefore, this script can have your api-token. If you
# re-generate your API Token after you download this script, the download will
# fail. If that happens, you can either re-download the script or you can edit
# this script replacing the old API Token with the new one. View your API token
# by going to "Account Home":
#
# https://www.earthsystemgrid.org/account/user/account-home.html
#
# and clicking on the "API Token" link under "Personal Account". You will be asked
# to log into the application before you can view your API Token.
#
# Usage: python3 python-ucar.cgd.ccsm4.geomip.ssp5.BWSSP585cmip6.15C.001.atm.proc.monthly_ave-20240202T0758.py
# Version: 0.1.2-alpha
#
# Dataset
# ucar.cgd.ccsm4.geomip.ssp5.BWSSP585cmip6.15C.001.atm.proc.monthly_ave
# 794afa6c-eae3-4209-946a-446789a0d6d5
# https://www.earthsystemgrid.org/dataset/ucar.cgd.ccsm4.geomip.ssp5.BWSSP585cmip6.15C.001.atm.proc.monthly_ave.html
# https://www.earthsystemgrid.org/dataset/id/794afa6c-eae3-4209-946a-446789a0d6d5.html
#
# Dataset Version
# 1
# c4d46697-d57c-4b55-96dc-54d824982ed3
# https://www.earthsystemgrid.org/dataset/ucar.cgd.ccsm4.geomip.ssp5.BWSSP585cmip6.15C.001.atm.proc.monthly_ave/version/1.html
# https://www.earthsystemgrid.org/dataset/version/id/c4d46697-d57c-4b55-96dc-54d824982ed3.html
#
################################################################

print('This Python 3 download script is experimental.  Please email feedback to esg-support@earthsystemgrid.org.\n')

args = {}
args.update({'apiToken': 'ocTWrqh3OlIho8oDkIBumx6lYu1vNSjQwpoKS340'})
args.update({'userAgent': 'python/{}/gateway/{}'.format(python_version(), '4.4.1-20240110-171341')})
args.update({'attemptMax': 10})
args.update({'initialSleepSeconds': 10})
args.update({'sleepMultiplier': 3})
args.update({'sleepMaxSeconds': 900})

data = [
     {'url':'https://tds.ucar.edu/thredds/fileServer/datazone/cdg/data/GeoMIP-SSP5/b.e21.BWSSP585cmip6.f09_g17.CMIP6-SSP5-8.5-WACCM.feedback.15C.001/atm/proc/tseries/month_1/b.e21.BWSSP585cmip6.f09_g17.CMIP6-SSP5-8.5-WACCM.feedback.15C.001.cam.h0.TH.201901-206812.nc','filename':'b.e21.BWSSP585cmip6.f09_g17.CMIP6-SSP5-8.5-WACCM.feedback.15C.001.cam.h0.TH.201901-206812.nc','bytes':'4759578229','md5Checksum':''},
     {'url':'https://tds.ucar.edu/thredds/fileServer/datazone/cdg/data/GeoMIP-SSP5/b.e21.BWSSP585cmip6.f09_g17.CMIP6-SSP5-8.5-WACCM.feedback.15C.001/atm/proc/tseries/month_1/b.e21.BWSSP585cmip6.f09_g17.CMIP6-SSP5-8.5-WACCM.feedback.15C.001.cam.h0.TH.206901-210012.nc','filename':'b.e21.BWSSP585cmip6.f09_g17.CMIP6-SSP5-8.5-WACCM.feedback.15C.001.cam.h0.TH.206901-210012.nc','bytes':'3050558456','md5Checksum':''},
     {'url':'https://tds.ucar.edu/thredds/fileServer/datazone/cdg/data/GeoMIP-SSP5/b.e21.BWSSP585cmip6.f09_g17.CMIP6-SSP5-8.5-WACCM.feedback.15C.001/atm/proc/tseries/month_1/b.e21.BWSSP585cmip6.f09_g17.CMIP6-SSP5-8.5-WACCM.feedback.15C.001.cam.h0.TREFHT.201901-206812.nc','filename':'b.e21.BWSSP585cmip6.f09_g17.CMIP6-SSP5-8.5-WACCM.feedback.15C.001.cam.h0.TREFHT.201901-206812.nc','bytes':'74195312','md5Checksum':''},
     {'url':'https://tds.ucar.edu/thredds/fileServer/datazone/cdg/data/GeoMIP-SSP5/b.e21.BWSSP585cmip6.f09_g17.CMIP6-SSP5-8.5-WACCM.feedback.15C.001/atm/proc/tseries/month_1/b.e21.BWSSP585cmip6.f09_g17.CMIP6-SSP5-8.5-WACCM.feedback.15C.001.cam.h0.TREFHT.206901-210012.nc','filename':'b.e21.BWSSP585cmip6.f09_g17.CMIP6-SSP5-8.5-WACCM.feedback.15C.001.cam.h0.TREFHT.206901-210012.nc','bytes':'47493522','md5Checksum':''},
     {'url':'https://tds.ucar.edu/thredds/fileServer/datazone/cdg/data/GeoMIP-SSP5/b.e21.BWSSP585cmip6.f09_g17.CMIP6-SSP5-8.5-WACCM.feedback.15C.001/atm/proc/tseries/month_1/b.e21.BWSSP585cmip6.f09_g17.CMIP6-SSP5-8.5-WACCM.feedback.15C.001.cam.h0.U.201901-206812.nc','filename':'b.e21.BWSSP585cmip6.f09_g17.CMIP6-SSP5-8.5-WACCM.feedback.15C.001.cam.h0.U.201901-206812.nc','bytes':'6613393985','md5Checksum':''},
     {'url':'https://tds.ucar.edu/thredds/fileServer/datazone/cdg/data/GeoMIP-SSP5/b.e21.BWSSP585cmip6.f09_g17.CMIP6-SSP5-8.5-WACCM.feedback.15C.001/atm/proc/tseries/month_1/b.e21.BWSSP585cmip6.f09_g17.CMIP6-SSP5-8.5-WACCM.feedback.15C.001.cam.h0.U.206901-210012.nc','filename':'b.e21.BWSSP585cmip6.f09_g17.CMIP6-SSP5-8.5-WACCM.feedback.15C.001.cam.h0.U.206901-210012.nc','bytes':'4232876501','md5Checksum':''},
     {'url':'https://tds.ucar.edu/thredds/fileServer/datazone/cdg/data/GeoMIP-SSP5/b.e21.BWSSP585cmip6.f09_g17.CMIP6-SSP5-8.5-WACCM.feedback.15C.001/atm/proc/tseries/month_1/b.e21.BWSSP585cmip6.f09_g17.CMIP6-SSP5-8.5-WACCM.feedback.15C.001.cam.h0zm.T.201901-206812.nc','filename':'b.e21.BWSSP585cmip6.f09_g17.CMIP6-SSP5-8.5-WACCM.feedback.15C.001.cam.h0zm.T.201901-206812.nc','bytes':'19981698','md5Checksum':''},
     {'url':'https://tds.ucar.edu/thredds/fileServer/datazone/cdg/data/GeoMIP-SSP5/b.e21.BWSSP585cmip6.f09_g17.CMIP6-SSP5-8.5-WACCM.feedback.15C.001/atm/proc/tseries/month_1/b.e21.BWSSP585cmip6.f09_g17.CMIP6-SSP5-8.5-WACCM.feedback.15C.001.cam.h0zm.T.206901-210012.nc','filename':'b.e21.BWSSP585cmip6.f09_g17.CMIP6-SSP5-8.5-WACCM.feedback.15C.001.cam.h0zm.T.206901-210012.nc','bytes':'12833507','md5Checksum':''},
     {'url':'https://tds.ucar.edu/thredds/fileServer/datazone/cdg/data/GeoMIP-SSP5/b.e21.BWSSP585cmip6.f09_g17.CMIP6-SSP5-8.5-WACCM.feedback.15C.001/atm/proc/tseries/month_1/b.e21.BWSSP585cmip6.f09_g17.CMIP6-SSP5-8.5-WACCM.feedback.15C.001.cam.h0zm.TREFHT.201901-206812.nc','filename':'b.e21.BWSSP585cmip6.f09_g17.CMIP6-SSP5-8.5-WACCM.feedback.15C.001.cam.h0zm.TREFHT.201901-206812.nc','bytes':'606107','md5Checksum':''},
     {'url':'https://tds.ucar.edu/thredds/fileServer/datazone/cdg/data/GeoMIP-SSP5/b.e21.BWSSP585cmip6.f09_g17.CMIP6-SSP5-8.5-WACCM.feedback.15C.001/atm/proc/tseries/month_1/b.e21.BWSSP585cmip6.f09_g17.CMIP6-SSP5-8.5-WACCM.feedback.15C.001.cam.h0zm.TREFHT.206901-210012.nc','filename':'b.e21.BWSSP585cmip6.f09_g17.CMIP6-SSP5-8.5-WACCM.feedback.15C.001.cam.h0zm.TREFHT.206901-210012.nc','bytes':'429430','md5Checksum':''},]

def main(args, data):

    for d in data:
        executeDownload(Download(args, d))

def executeDownload(download):

    if not os.path.isfile(download.filename):
        attemptAndValidateDownload(download)
        moveDownload(download)
    else:
        download.success = True
        download.valid = True

    reportDownload(download)

def moveDownload(download):

    if download.success and (download.valid or download.vwarning):
        os.rename(download.filenamePart, download.filename)

def reportDownload(download):

    if download.success and download.valid:
        print('successfully downloaded {}'.format(download.filename))

    if download.success and not download.valid and download.vwarning:
        print('downloaded with warning {}: {}'.format(download.filename, download.vwarning))

    if download.success and not download.valid and download.verror:
        print('downloaded with validation error {}: {}'.format(download.filename, download.verror))

    if not download.success and download.error:
        print('download error {}: {}'.format(download.filename, download.error))

def attemptAndValidateDownload(download):

    while download.attempt:
        downloadFile(download)

    if download.success:
        validateFile(download)

def downloadFile(download):

    try :
        startOrResumeDownload(download)
    except HTTPError as error:
        handleHTTPErrorAttempt(download, error)
    except URLError as error:
        handleRecoverableAttempt(download, error)
    except TimeoutError as error:
        handleRecoverableAttempt(download, error)
    except Exception as error:
        handleIrrecoverableAttempt(download, error)
    else:
        handleSuccessfulAttempt(download)

def startOrResumeDownload(download):

    if os.path.isfile(download.filenamePart):
        resumeDownloadFile(download)
    else:
        startDownloadFile(download)

def resumeDownloadFile(download):

    print('resuming download of {}'.format(download.filename))
    opener = createOpener(createResumeHeaders(download))
    readFile(download, opener)

def startDownloadFile(download):

    print('starting download of {}'.format(download.filename))
    opener = createOpener(createStartHeaders(download))
    readFile(download, opener)

def createResumeHeaders(download):

    headers = createStartHeaders(download)
    headers.append(createRangeHeader(download))

    return headers

def createStartHeaders(download):

    headers = []
    headers.append(createUserAgentHeader(download))

    if download.apiToken:
        headers.append(createAuthorizationHeader(download))

    return headers

def createUserAgentHeader(download):

    return ('User-agent', download.userAgent)

def createAuthorizationHeader(download):

    return ('Authorization', 'api-token {}'.format(download.apiToken))

def createRangeHeader(download):

    start = os.path.getsize(download.filenamePart)
    header = ('Range', 'bytes={}-'.format(start))

    return header

def createOpener(headers):

    opener = urllib.request.build_opener()
    opener.addheaders = headers

    return opener

def readFile(download, opener):

    with opener.open(download.url) as response, open(download.filenamePart, 'ab') as fh:
        collectResponseHeaders(download, response)
        shutil.copyfileobj(response, fh)

def collectResponseHeaders(download, response):

    download.responseHeaders = response.info()
    if download.responseHeaders.get('ETag'):
        download.etag = download.responseHeaders.get('ETag').strip('"')

def handleHTTPErrorAttempt(download, httpError):

    if httpError.code == 416: # 416 is Range Not Satisfiable
        # likely the file completely downloaded and validation was interrupted,
        # therefore calling it successfully downloaded and allowing validation
        # to say otherwise
        handleSuccessfulAttempt(download)
    else:
        handleRecoverableAttempt(download, httpError)

def handleRecoverableAttempt(download, error):

    print('failure on attempt {} downloading {}: {}'.format(download.attemptNumber, download.filename, error))

    if download.attemptNumber < download.attemptMax:
        sleepBeforeNextAttempt(download)
        download.attemptNumber += 1
    else:
        handleIrrecoverableAttempt(download, error)

def sleepBeforeNextAttempt(download):

    sleepSeconds = download.initialSleepSeconds * (download.sleepMultiplier ** (download.attemptNumber - 1))

    if sleepSeconds > download.sleepMaxSeconds:
        sleepSeconds = download.sleepMaxSeconds

    print('sleeping for {} seconds before next attempt'.format(sleepSeconds))
    time.sleep(sleepSeconds)

def handleIrrecoverableAttempt(download, error):

    download.attempt = False
    download.error = error

def handleSuccessfulAttempt(download):

    download.attempt = False
    download.success = True

def validateFile(download):

    try:
        validateAllSteps(download)
    except InvalidDownload as error:
        download.valid = False
        download.vwarning = str(error)
    except Exception as error:
        download.valid = False
        download.verror = error
    else:
        download.valid = True

def validateAllSteps(download):

    verrorData = validatePerData(download)
    verrorEtag = validatePerEtag(download)
    verrorStale = validateStaleness(download)

    if verrorData and verrorEtag:
        raise verrorData

    if verrorStale:
        raise verrorStale

def validatePerData(download):

    try:
        validateBytes(download)
        validateChecksum(download)
    except InvalidDownload as error:
        return error
    else:
        return None

def validateBytes(download):

    size = os.path.getsize(download.filenamePart)
    if not download.bytes == size:
        raise InvalidSizeValue(download, size)

def validateChecksum(download):

    if download.md5Checksum:
        md5Checksum = readMd5Checksum(download)
        if not download.md5Checksum == md5Checksum:
            raise InvalidChecksumValue(download, md5Checksum)
    else:
        raise UnableToPerformChecksum(download)

def readMd5Checksum(download):

    hash_md5 = hashlib.md5()

    with open(download.filenamePart, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()

def validatePerEtag(download):

    try:
        validateChecksumEtag(download)
    except InvalidDownload as error:
        return error
    else:
        return None

def validateChecksumEtag(download):

    if isEtagChecksum(download):
        md5Checksum = readMd5Checksum(download)
        if not download.etag == md5Checksum:
            raise InvalidChecksumValuePerEtag(download, md5Checksum)
    else:
        raise UnableToPerformChecksum(download)

def isEtagChecksum(download):

    return download.etag and re.fullmatch(r'[a-z0-9]+', download.etag)

def validateStaleness(download):

    try:
        validateStaleChecksum(download)
    except InvalidDownload as error:
        return error
    else:
        return None

def validateStaleChecksum(download):

    if isEtagChecksum(download):
        if not download.md5Checksum or download.md5Checksum != download.etag:
            raise StaleChecksumValue(download)

class InvalidDownload(Exception):

    pass

class InvalidSizeValue(InvalidDownload):

    def __init__(self, download, actual):
        super().__init__('invalid byte size for {}: expected {}, downloaded file {}'.format(download.filename, download.bytes, actual))

class InvalidChecksumValue(InvalidDownload):

    def __init__(self, download, actual):
        super().__init__('invalid md5 checksum for {}: expected {} (see data), downloaded file {}'.format(download.filename, download.md5Checksum, actual))

class InvalidChecksumValuePerEtag(InvalidDownload):

    def __init__(self, download, actual):
        super().__init__('invalid md5 checksum for {}: expected {} (see server etag), downloaded file {}'.format(download.filename, download.etag, actual))

class UnableToPerformChecksum(InvalidDownload):

    def __init__(self, download):
        super().__init__('cannot verify md5 checksum of {}'.format(download.filename))

class StaleChecksumValue(InvalidDownload):

    def __init__(self, download):
        md5Checksum = 'none' if not download.md5Checksum else download.md5Checksum
        super().__init__('stale md5 checksum value for file {}: script {}, server etag {}'.format(download.filename, md5Checksum, download.etag))

class Download():

    def __init__(self, args, datum):

        self.apiToken = args.get('apiToken')
        self.userAgent = args.get('userAgent')
        self.attemptMax = args.get('attemptMax')
        self.initialSleepSeconds = args.get('initialSleepSeconds')
        self.sleepMultiplier = args.get('sleepMultiplier')
        self.sleepMaxSeconds = args.get('sleepMaxSeconds')

        self.url = datum.get('url')
        self.filename = datum.get('filename')
        self.bytes = int(datum.get('bytes'))
        self.md5Checksum = datum.get('md5Checksum')

        self.filenamePart = self.filename + '.part'
        self.success = False
        self.attempt = True
        self.attemptNumber = 1
        self.responseHeaders = {}
        self.etag = None
        self.error = None
        self.valid = False
        self.vwarning = None
        self.verror = None

    def __str__(self):
        return f'url: {self.url}, filename: {self.filename}, bytes: {self.bytes}, md5Checksum: {self.md5Checksum}'

if __name__ == '__main__':
    main(args, data)
