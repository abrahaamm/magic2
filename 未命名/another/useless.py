from tqdm import tqdm

from .understand import Understand


class Useless(Understand):
    def __init__(self, source, target):
        super().__init__(source, target)

    def understand(self):
        sources_and_targets = tqdm(zip(self.source, self.target), desc='Useless', total=len(self.source))
        for source, target in sources_and_targets:
            print(f"understand {source} file to {target}")
            events = self.json_read(source)
            events = self.understand_events(events)
            self.json_write(target, events)

    @staticmethod
    def understand_events(events):
        """ delete useless events 
        Args: 
            events(list) : list of json type event 
        
        Returns: 
            list(events) : list of events after deleting useless events 

        """
        events = [
            event for event in events if
            # No identical information such as uuid
            event['sbj'] is not None and 
            event['sbj+'] is not None and 
            event['obj'] is not None and 
            event['obj+'] is not None and
            event['obj'] not in ('ProvenanceTagNode', 'Principal', 'SrcSinkObject', 'UnnamedPipeObject') and
            # Memory is too fine-grained
            event['obj+'] != 'MEMORY' and
            event['rel'] not in ('MMAP', 'MPROTECT', 'OTHER', 'CORRELATE', 'SHM') and
            # Only need to focus on the actions between OPEN and CLOSE (very common to ignore)
            event['rel'] not in ('OPEN', 'CLOSE')
        ]

        # Remove port in the FROM of netflow
        for event in events:
            if event['obj+'] == 'NETFLOW':
                from_, to = event['obj'].split('>>')
                from_split = from_.replace(':', '.').split('.')
                if len(from_split) in (5, 9):
                    from_split = from_split[:-1]
                    if len(from_split) == 4:
                        from_ = '.'.join(from_split)
                    else:
                        from_ = ':'.join(from_split)
                    event['obj'] = f'{from_}>>{to}'
        return events
