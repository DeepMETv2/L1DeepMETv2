from coffea.nanoevents import NanoEventsFactory
from coffea.nanoevents.schemas import NanoAODSchema,BaseSchema
import numpy as np
import os
from optparse import OptionParser
import concurrent.futures
import glob
import awkward as ak
import time
import json
from collections import OrderedDict,defaultdict
#recdd = lambda : defaultdict(recdd) ## define recursive defaultdict

JSON_LOC = 'filelist.json'

'''

Changes made from DeepMETv2
1. genmet_list contains only genMet (in both future_savez and __main__)
2. particle_list contains L1PuppiCands information instead of PFCands, based on L1MET's input (in both future_savez and __main__)
3. datasetsname replaces Znunu to TTbar (in __main__); also change the JSON file used
4. nparticles_per_event based on n(L1PuppiCands) in an event instead of PFCands (in __main__)

'''


def multidict_tojson(filepath, indict):
    ## expand into multidimensional dictionary
    with open(filepath, "w") as fo:
        json.dump( indict, fo)
        print("save to %s" %filepath)

def future_savez(i, tot):
        #tic=time.time()
        genmet_list = [
                events.genMet.pt[i] * np.cos(events.genMet.phi[i]),
                events.genMet.pt[i] * np.sin(events.genMet.phi[i]),
        ]
      
        # particle_list follows L89-94 in https://github.com/jmduarte/L1METML/blob/main/convertNanoToHDF5_L1triggerToDeepMET.py
        particle_list = np.column_stack([
                        events.L1PuppiCands.pt[i],
                        events.L1PuppiCands.eta[i],
                        events.L1PuppiCands.phi[i],
                        events.L1PuppiCands.puppiWeight[i],
                        events.L1PuppiCands.pdgId[i],
                        events.L1PuppiCands.charge[i],
        ])

        eventi = [particle_list,genmet_list]
        #toc=time.time()
        #print(toc-tic)
        return eventi


if __name__ == '__main__':
        
        parser = OptionParser()
        parser.add_option('-d', '--dataset', help='dataset', default='TTbar', dest='dataset')  # Make TTbar the default option for now
        parser.add_option('-s', '--startfile',type=int, default=0, help='startfile')
        parser.add_option('-e', '--endfile',type=int, default=1, help='endfile')
        (options, args) = parser.parse_args()

        datasetsname = {
            'TTbar': [ 'perfNano_TTbar_PU200.root' ]
        }

        # Be nice to eos, save list to a file
        #filelists = recdd()
        #for datset in datasetsname.keys():
        #    filelists[datset] = glob.glob('/eos/uscms/store/group/lpcjme/NanoMET/'+datasetsname[datset][0]+'/*/*/*/*root')
        #    filelists[datset] = [x.replace('/eos/uscms','root://cmseos.fnal.gov/') for x in filelists[datset] ]
        #multidict_tojson(JSON_LOC, filelists )
        #exit()
        
        dataset=options.dataset
        if dataset not in datasetsname.keys():
            print('choose one of them: ', datasetsname.keys())
            exit()
        
        #Read file from json
        with open(JSON_LOC, "r") as fo:
            file_names = json.load(fo)
        file_names = file_names[dataset]
        print('find ', len(file_names)," files")

        if options.startfile>=options.endfile and options.endfile!=-1:
            print("make sure options.startfile<options.endfile")
            exit()

        inpz=0
        eventperfile=1000
        currentfile=0

        for ifile in file_names:
            if currentfile<options.startfile:
                currentfile+=1
                continue
            events = NanoEventsFactory.from_root(ifile, schemaclass=NanoAODSchema).events()
            nevents_total = len(events)
            print(ifile, ' Number of events:', nevents_total)
            
            for i in range(int(nevents_total / eventperfile)+1):
                if i< int(nevents_total / eventperfile):
                    print('from ',i*eventperfile, ' to ', (i+1)*eventperfile)
                    events_slice = events[i*eventperfile:(i+1)*eventperfile]
                elif i == int(nevents_total / eventperfile) and i*eventperfile<=nevents_total:
                    print('from ',i*eventperfile, ' to ', nevents_total)
                    events_slice = events[i*eventperfile:nevents_total]
                else:
                    print(' weird ... ')

                nparticles_per_event = max(ak.num(events_slice.L1PuppiCands.pt, axis=1))
                print("max nL1PuppiCands in this range: ", nparticles_per_event)
                tic=time.time()
                met_list = np.column_stack([
                        events_slice.genMet.pt * np.cos(events_slice.genMet.phi),
                        events_slice.genMet.pt * np.sin(events_slice.genMet.phi),
                ])
                particle_list = ak.concatenate([
                             [ ak.fill_none(ak.pad_none(events_slice.L1PuppiCands.pt, nparticles_per_event, clip=True), -999)           ],
                             [ ak.fill_none(ak.pad_none(events_slice.L1PuppiCands.eta, nparticles_per_event, clip=True), -999)          ],
                             [ ak.fill_none(ak.pad_none(events_slice.L1PuppiCands.phi, nparticles_per_event, clip=True), -999)          ],
                             [ ak.fill_none(ak.pad_none(events_slice.L1PuppiCands.puppiWeight, nparticles_per_event, clip=True), -999)  ],
                             [ ak.fill_none(ak.pad_none(events_slice.L1PuppiCands.pdgId, nparticles_per_event, clip=True), -999)        ],
                             [ ak.fill_none(ak.pad_none(events_slice.L1PuppiCands.charge, nparticles_per_event, clip=True), -999)       ],
                ])

                outdir = os.environ['PWD'] + '/raw/'
                os.system('mkdir -p ' + outdir)

                npz_file = outdir + dataset+'_file'+str(currentfile)+'_slice_'+str(i)+'_nevent_'+str(len(events_slice))

                #npz_file=os.environ['PWD']+'/raw/'+dataset+'_file'+str(currentfile)+'_slice_'+str(i)+'_nevent_'+str(len(events_slice))
                np.savez(npz_file, x=np.array(particle_list), y=np.array(met_list)) 

                toc=time.time()
                print('time:',toc-tic)
            currentfile+=1
            if currentfile>=options.endfile:
                print('=================> finished ')
                exit()
