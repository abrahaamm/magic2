import json
import os
from collections import defaultdict
from datetime import datetime

import pytz
from tqdm import tqdm

from .data import Data
import hashlib

def _hash_string(input_string, algorithm='sha256'): 
    hash_object = hashlib.new(algorithm) 
    hash_object.update(input_string.encode('utf-8'))
    hex_dig = hash_object.hexdigest()
    return hex_dig

class DARPADict(dict):
    def __init__(self):
        super().__init__()
        self.conflict = defaultdict(set)

    def __setitem__(self, key, value):
        value = None if value == '' else value
        if key in self and value != self[key]:
            self.conflict[key].add(value)
            self.conflict[key].add(self[key])
        super().__setitem__(key, value)


class DARPA(Data):
    def __init__(self, data_list, provenance_dir, dataset_name, dataset_dir):
        super().__init__(data_list, provenance_dir, dataset_name, dataset_dir)

        self.PREFIX = 'com.bbn.tc.schema.avro.cdm18.'
        self.TIMEZONE = pytz.timezone('US/Eastern')
        self.PLUS = '>>>>'
        self.DELIMITER = '><'
        self.FLOW = '>>'

        self.inputs, self.outputs = self.inputs_and_outputs()
        self.uuid2entity = self.uuid2entity()

    def load_and_dump(self):
        """ unify original darpa dataset to clean-named right-ordered files(provenance graph)

        Args : None 

        Returns : None 

        Files : 
            Write '0_1.json', '0_2.json', ... (These are the names of unified dataset files) in self.provenance_dir
        """
        input_and_output = tqdm(zip(self.inputs, self.outputs), total=len(self.inputs), desc='Unify')
        for input_, output in input_and_output:
            print(f"Unifying file : {input_} to file {output}... ") 
            fr = open(input_, 'r', encoding='utf-8')
            fw = open(output, 'w', encoding='utf-8')
            lines_count = 0
            parsed_count = 0
            try:
                # jsonl
                for line in fr:
                    lines_count += 1
                    entry = json.loads(line)['datum']
                    if f'{self.PREFIX}Event' not in entry:
                        continue
                    event = self.unify(entry[f'{self.PREFIX}Event'])
                    parsed_count += 1
                    fw.write(json.dumps(event) + '\n')
            except json.decoder.JSONDecodeError:
                # json
                for line in fr:
                    lines_count += 1
                    entry = json.loads(line.strip()[:-1])['datum']
                    if f'{self.PREFIX}Event' not in entry:
                        continue
                    parsed_count += 1
                    event = self.unify(entry[f'{self.PREFIX}Event'])
                    fw.write(json.dumps(event) + '\n')
            print(f"All lines : {lines_count}, parsed lines : {parsed_count}")
            fr.close()
            fw.close()

    def unify(self, event):
        def plus(entity):
            if entity is None or self.PLUS not in entity:
                return None, entity
            else:
                return entity.split(self.PLUS)

        sid = event['subject'][f'{self.PREFIX}UUID'] if event['subject'] is not None else None
        rid = event['uuid']
        oid = event['predicateObject'][f'{self.PREFIX}UUID'] if event['predicateObject'] is not None else None
        hid = event['hostId']

        sbj_, sbj = plus(self.uuid2entity.get(sid, None))
        rel_, rel = plus(self.uuid2entity.get(rid, None))
        obj_, obj = plus(self.uuid2entity.get(oid, None))

        unified_event = {
            'id': self._unified_id,

            't': datetime.fromtimestamp(event['timestampNanos'] / 1000000000).astimezone(self.TIMEZONE).strftime(
                '%Y-%m-%d %H:%M:%S:%f'),

            'sbj': sbj,
            'rel': rel,
            'obj': obj,

            'sid': sid,
            'rid': rid,
            'oid': oid,
            'hid': hid,

            'sbj+': sbj_,
            'rel+': rel_,
            'obj+': obj_,
        }
        self._unified_id += 1
        return unified_event

    def inputs_and_outputs(self):
        """ parse input data files and output files 

        Returns: 
            inputs: all input files' names. e.g. ta1-trace-e3-official.json.114 
            outputs: all output files' names. e.g. test/2.jsonl
        """
        inputs = [os.path.join(self.dataset_dir, dataset_file) for dataset_file in self.data_list]
        outputs = [os.path.join(self.provenance_dir, f'{index}.jsonl') for index, _ in enumerate(self.data_list)]
        return inputs, outputs

    # Below is uuid -> entity
    def uuid2entity(self):
        uuid2entity_path = os.path.join('./data/uuid2entity/', f'{_hash_string(self.dataset_dir)}_uuid2entity.tsv')
        if os.path.exists(uuid2entity_path):
            return self.load_uuid2entity(uuid2entity_path)
        else:
            return self.generate_uuid2entity(uuid2entity_path)

    @staticmethod
    def load_uuid2entity(uuid2entity_path):
        uuid2entity = DARPADict()
        with open(uuid2entity_path, 'r', encoding='utf-8') as f:
            for line in f:
                uuid, entity = line.strip().split('\t')
                if entity == 'None':
                    entity = None
                uuid2entity[uuid] = entity
        return uuid2entity

    def generate_uuid2entity(self, uuid2entity_path=None):
        """ Generate uuid to entity dict file 
        
        Args: 
            uuid2entity_path(str): the file name for uuid2entity dict. Default None, usually uuid2entity.tsv 
        
        Returns: 
            DARPADict() : the uuid2entity dict for all inputs' files. 
        
        Files: 
            write uuid2entity dict to the file 'uuid2eneity_path' in the form of {uuid}\t{entity}\n
        """
        uuid2entity = DARPADict()
        all_dataset_files = [os.path.join(self.dataset_dir, data_file) for data_file in os.listdir(self.dataset_dir)]
        for file in tqdm(all_dataset_files, desc=f'Generate uuid2entity'):
            with open(file, 'r', encoding='utf-8') as f:
                try:
                    # jsonl
                    for line in f:
                        entry = json.loads(line)['datum']
                        try:
                            self.append_uuid2entity(entry, uuid2entity)
                        except ValueError:
                            self.PREFIX = 'com.bbn.tc.schema.avro.cdm18.' if self.PREFIX == '' else ''
                            self.append_uuid2entity(entry, uuid2entity)
                except json.decoder.JSONDecodeError:
                    # json
                    for line in f:
                        entry = json.loads(line.strip()[:-1])['datum']
                        try:
                            self.append_uuid2entity(entry, uuid2entity)
                        except ValueError:
                            self.PREFIX = 'com.bbn.tc.schema.avro.cdm18.' if self.PREFIX == '' else ''
                            self.append_uuid2entity(entry, uuid2entity)
            print(file, len(uuid2entity.conflict))

        # Update using existing uuids
        for uuid, entity in uuid2entity.items():
            if entity.startswith(f'UUID{self.PLUS}'):
                uuid2entity[uuid] = uuid2entity.get(entity.replace(f'UUID{self.PLUS}', ''), uuid2entity[uuid])

        if uuid2entity_path is not None:
            with open(uuid2entity_path, 'w', encoding='utf-8') as f:
                for uuid, entity in uuid2entity.items():
                    f.write(f'{uuid}\t{entity}\n')
        return uuid2entity

    def append_uuid2entity(self, entry, uuid2entity):
        """ append uuid and entity's information to uuid2entity dict 
        Args: 
            entry(json) : darpa json object, one entity 

        Returns: None 

        Function: 
            append uuid2entity['uuid'] = entity's usefule information      
        """
        if f'{self.PREFIX}Event' in entry:
            x = entry[f'{self.PREFIX}Event']
            uuid2entity[x['uuid']] = x['type'].replace('EVENT_', f'EVENT{self.PLUS}')

        # All entries below has specific information
        elif f'{self.PREFIX}Subject' in entry:
            x = entry[f'{self.PREFIX}Subject']
            try:
                if x['type'] == 'SUBJECT_UNIT':
                    uuid2entity[x['uuid']] = f"UNIT{self.PLUS}{x['properties']['map']['name']}"
                elif x['type'] == 'SUBJECT_PROCESS':
                    if 'name' in x['properties']['map']:
                        uuid2entity[x['uuid']] = f"PROCESS{self.PLUS}{x['properties']['map']['name']}"
                    elif 'host' in x['properties']['map']:
                        uuid2entity[x['uuid']] = f"UUID{self.PLUS}{x['properties']['map']['host'].upper()}"
                    else:
                        raise KeyError('No identified key.')
                else:
                    uuid2entity[x['uuid']] = f"SUBJECT{self.PLUS}{x['properties']['map']['name']}"
            except (KeyError, TypeError):
                uuid2entity[x['uuid']] = f"UUID{self.PLUS}{x['uuid']}"
        elif f'{self.PREFIX}FileObject' in entry:
            x = entry[f'{self.PREFIX}FileObject']
            try:
                if x['type'] == 'FILE_OBJECT_DIR':
                    uuid2entity[x['uuid']] = f"DIR{self.PLUS}{x['baseObject']['properties']['map']['path']}"
                else:
                    uuid2entity[x['uuid']] = f"FILE{self.PLUS}{x['baseObject']['properties']['map']['path']}"
            except (KeyError, TypeError):
                uuid2entity[x['uuid']] = f"UUID{self.PLUS}{x['uuid']}"
        elif f'{self.PREFIX}NetFlowObject' in entry:
            x = entry[f'{self.PREFIX}NetFlowObject']
            uuid2entity[x['uuid']] = (
                f'NETFLOW{self.PLUS}{x["localAddress"]}:{x["localPort"]}'
                f'{self.FLOW}{x["remoteAddress"]}:{x["remotePort"]}')
        elif f'{self.PREFIX}IpcObject' in entry:
            x = entry[f'{self.PREFIX}IpcObject']
            uuid2entity[x['uuid']] = f"IPC{self.PLUS}{x['properties']['map']['path']}"
        elif f'{self.PREFIX}MemoryObject' in entry:
            x = entry[f'{self.PREFIX}MemoryObject']
            uuid2entity[x['uuid']] = f'MEMORY{self.PLUS}{x["memoryAddress"]}'
        elif f'{self.PREFIX}Host' in entry:
            x = entry[f'{self.PREFIX}Host']
            ips = [(z if '.' in z else ':'.join(o.zfill(4) for o in z.split('%')[0].split(':')))  # ipv6
                   for y in x['interfaces'] for z in y['ipAddresses']]
            ips = sorted(ips, key=lambda ip: len(ip.replace(':', '.').split('.')))
            ips = self.DELIMITER.join(ips)
            uuid2entity[x['uuid']] = f"HOST{self.PLUS}{ips}"

        # All entries below has no specific information
        elif f'{self.PREFIX}ProvenanceTagNode' in entry:
            x = entry[f'{self.PREFIX}ProvenanceTagNode']
            uuid2entity[x['tagId']] = 'ProvenanceTagNode'
        elif f'{self.PREFIX}Principal' in entry:
            x = entry[f'{self.PREFIX}Principal']
            uuid2entity[x['uuid']] = 'Principal'
        elif f'{self.PREFIX}SrcSinkObject' in entry:
            x = entry[f'{self.PREFIX}SrcSinkObject']
            uuid2entity[x['uuid']] = 'SrcSinkObject'
        elif f'{self.PREFIX}UnnamedPipeObject' in entry:
            x = entry[f'{self.PREFIX}UnnamedPipeObject']
            uuid2entity[x['uuid']] = 'UnnamedPipeObject'
        elif any(key in entry for key in
                 [f'{self.PREFIX}TimeMarker', f'{self.PREFIX}StartMarker', f'{self.PREFIX}EndMarker',
                  f'{self.PREFIX}RegistryKeyObject', f'{self.PREFIX}UnitDependency']):
            pass

        # Entries that cannot handle
        else:
            raise ValueError(f'Cannot handle {entry}')
