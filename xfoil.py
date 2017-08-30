# -*- coding: utf-8 -*-

import os
import numpy as np
import subprocess as sp
import re


def polar(afile, re, *args,**kwargs):
    """Berechnet Profilpolaren und liest diese ein.
    
    :param afile: Pfad zu Profilkoordinatendatei oder NACA4 Code
    :type afile: string
    :param re: Reynoldszahl
    :type re: int
    :param alfaseq: Sequenz von Anstellwinkeln
    :type alfaseq: iterateable
    :rtype: dict
    """
    calcPolar(afile, re,'polar.txt', *args, **kwargs)
    data = readPolar('polar.txt')
    #raw_input('EnterKex')
    deletePolar('polar.txt')
    return data


def calcPolar(afile, re, polarfile, alfaseq=[], clseq=[], refine=False, max_iter=200, n=None):
    """Führt XFOIL aus und lässt Profilpolaren generieren.
    
    :param afile: Pfad zu Profilkoordinatendatei oder NACA4 Code
    :type afile: string
    :param re: Reynoldszahl
    :type re: int
    :param polarfile: Ausgabedatei für XFOIL, darf nicht existieren
    :param alfaseq: Sequenz von Anstellwinkeln
    :type alfaseq: iterateable
    :param clseq: Sequenz von Auftriebsbeiwerten, Alternativparameter zu alfaseq
    :type clseq: iterateable 
    :param refine: Soll XFOIL ein refinement des Profils durchführen
    :type refine: bool
    :param max_iter: Maximale Iterationsanzahl
    :type max_iter: int
    :param n: Grenzschichtparameter
    :type n: int
    :rtype: None
    """
    
    import subprocess as sp
    import numpy as np
    import sys,os
    
    if(os.name == 'posix'):
        xfoilbin = 'xfoil'
    elif(os.name == 'nt'):
        xfoilbin = 'xfoil.exe'
    else:
        print('Betriebssystem %s wird nicht unterstützt'%os.name)   
    
    
    pxfoil = sp.Popen([xfoilbin], stdin=sp.PIPE, stdout=None, stderr=None)
    
    def write2xfoil(string):
        if(sys.version_info > (3,0)):
            string = string.encode('ascii')
            
        pxfoil.stdin.write(string)
        
    if(afile.isdigit()):
        write2xfoil('NACA ' + afile + '\n')
    else:
        write2xfoil('LOAD ' + afile + '\n')
        
        if(refine):
            write2xfoil('GDES\n')
            write2xfoil('CADD\n')
            write2xfoil('\n')
            write2xfoil('\n')
            write2xfoil('\n')
            write2xfoil('X\n ')
            write2xfoil('\n')
            write2xfoil('PANEL\n')

    write2xfoil('OPER\n')
    if n != None:
        write2xfoil('VPAR\n')
        write2xfoil('N '+str(n)+'\n')
        write2xfoil('\n')
    write2xfoil('ITER '+str(max_iter)+'\n')
    write2xfoil('VISC\n')
    write2xfoil(str(re) + '\n')
    write2xfoil('PACC\n')
    write2xfoil('\n')
    write2xfoil('\n')
    for alfa in alfaseq:
        write2xfoil('A ' + str(alfa) + '\n')
    for cl in clseq:
        write2xfoil('CL ' + str(cl) + '\n')
    write2xfoil('PWRT 1\n')
    write2xfoil(polarfile + '\n')
    write2xfoil('\n')

    pxfoil.communicate(str('quit').encode('ascii'))

def readPolar(infile):
    """ Liest XFOIL-Polarendatei ein.
    
    :param infile: Polarendatei
    :rtype: dict
    """
    
    regex = re.compile('(?:\s*([+-]?\d*.\d*))')
    
    with open(infile) as f:
        lines = f.readlines()
        
        a           = []
        cl          = []
        cd          = []
        cdp         = []
        cm          = []
        xtr_top     = []
        xtr_bottom  = []
        
        
        for line in lines[12:]:
            linedata = regex.findall(line)
            a.append(float(linedata[0]))
            cl.append(float(linedata[1]))       
            cd.append(float(linedata[2]))
            cdp.append(float(linedata[3]))
            cm.append(float(linedata[4]))
            xtr_top.append(float(linedata[5]))
            xtr_bottom.append(float(linedata[6]))
            
        data = {'a': np.array(a), 'cl': np.array(cl) , 'cd': np.array(cd), 'cdp': np.array(cdp),
             'cm': np.array(cm), 'xtr_top': np.array(xtr_top), 'xtr_bottom': np.array(xtr_bottom)}
        
        return data


def deletePolar(infile):
    """ Löscht Datei. """
    os.remove(infile)

