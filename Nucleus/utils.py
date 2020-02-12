"""Gene calling utility classes and functions."""

import cv2
import json
import numpy as np
import os
import re
from bioutils import dnaEncoder, dna_reverse_complement
from datatoolkit import to_sparse_categorical

# TODO: Finish documentation
# TODO: Add comments
# TODO: Make ORF inherit from Feature

class Genome(object):
    """A Genome object facilitates reading and organizing contigs."""

    def __init__(self, gid=None, source=None):
        self.gid = gid
        self.source = source
        self.contigs = []
        # TODO: Change contigs to a dictionary

        if source is not None:
            if os.path.isfile(source+'.fna'):
                self.add_fasta(source+'.fna')
            if os.path.isfile(source+'.fsa'):
                self.add_fasta(source+'.fsa')

            if os.path.isfile(source+'.PATRIC.gff'):
                self.add_gff(source+'.PATRIC.gff')
            if os.path.isfile(source+'.gff'):
                self.add_gff(source+'.gff')

            if os.path.isfile(source+'.bam'):
                self.add_bam(source+'.bam')

    def __str__(self):
        rv = ''

        if self.gid is not None:
            rv += ' name=' + self.gid.__repr__()

        if self.source is not None:
            rv += ' source=' + self.source.__repr__()

        rv += '; ' + str(len(self.contigs)) + ' contigs'
        rv += f'; {sum(len(c.dnaPos) for c in self.contigs):,} bp; {100*self.gc:0.1f}% gc'

        return '<Genome' + rv + '>'

    def __repr__(self):
        return self.__str__()

    def get_contig_by_id(self, cid):
        for c in self.contigs:
            if c.cid == cid:
                return c
        return None

    def add_fasta(self, fastaFname):
        contig2dna = read_fasta(fastaFname)

        for contigId, dna in contig2dna.items():
            c = self.get_contig_by_id(contigId)
            if c is None:
                c = Contig(cid=contigId, genome=self)
                self.contigs.append(c)
            c.dnaPos = dna

    def add_gff(self, gffFname):
        contig2features = read_gff(gffFname)

        for contigId, features in contig2features.items():
            c = self.get_contig_by_id(contigId)
            if c is None:
                c = Contig(cid=contigId, genome=self)
                self.contigs.append(c)
            c.features.extend(features)

    def add_bam(self, bamFname):
        with pysam.AlignmentFile(bamFname, 'rb') as f:
            for contigId in sorted(f.references):
                # contigIdx = int(contigId.split('con.')[-1])
                # c = self.contigs[contigIdx-1]

                c = self.get_contig_by_id(contigId)
                l = f.lengths[f.references.index(contigId)]

                rnaProfile = np.zeros((l+1,))

                pile = f.pileup(contig=contigId)
                for x in pile:
                    rnaProfile[x.reference_pos] = min(x.nsegments, 100)

                c.rnaProfile = rnaProfile

    @property
    def gc(self):
        return sum(c.gc*len(c.dnaPos) for c in self.contigs) / sum(len(c.dnaPos) for c in self.contigs)

class Contig(object):
    """A Contig stores a sequence and features on that sequence."""

    def __init__(self, cid=None, genome=None):
        self.cid = cid
        self.genome = genome
        self._dnaPos = None
        self._dnaNeg = None
        self.gc = None
        self.rnaProfile = None
        self.coding = None
        self.features = []
        self.orfs = []

    def __str__(self):
        rv = ''

        if self.cid is not None:
            rv += ' name=' + self.cid.__repr__()

        if self.dnaPos is not None:
            rv += f'; {len(self.dnaPos):,} bp; {100*self.gc:0.1f}% gc'

        if self.rnaProfile is not None:
            rv += '; has RNA'

        rv += '; ' + str(len(self.features)) + ' features'
        rv += '; ' + str(len(self.orfs)) + ' orfs'

        return '<Contig' + rv + '>'

    def __repr__(self):
        return self.__str__()

    def annotate_coding(self):
        if not self.dna:
            raise AttributeError('Contig has no dna')
        if not self.features:
            raise AttributeError('Contig has no features')

        self.coding = np.zeros(len(self.dna))

        for feature in self.features:
            if feature.featureType != 'CDS':
                continue

            if feature.strand == '+':
                for frame, i in enumerate(range(feature.start, feature.stop)):
                    self.coding[i] = frame%3 + 1
            else:
                for frame, i in enumerate(range(feature.start, feature.stop)):
                    self.coding[i] = frame%3 - 3

    def find_orfs(self):
        self._find_orfs(self.dnaPos, '+')
        self._find_orfs(self.dnaNeg, '-')

    def _find_orfs(self, dna, strand):
        naL = [[], [], []]
        for match in re.finditer(r'(?=[agt]tg)', dna):
            loc = match.start()
            frame = loc%3
            naL[frame].append(loc)

        noL = [[0], [1], [2]]
        for match in re.finditer(r'(?=taa|tag|tga)', dna):
            loc = match.start()
            frame = loc%3
            noL[frame].append(loc+3)

        for frame in range(3):
            j = -1
            for i in range(len(noL[frame])-1):
                front = noL[frame][i]
                back = noL[frame][i+1]

                # Find all of the potential starts in this ORF to reduce computation later
                starts = []
                for j in range(j+1, len(naL[frame])):
                    # Once we are outside the ORF, stop searching
                    if naL[frame][j] >= back:
                        j -= 1
                        break
                    starts.append(naL[frame][j])

                if len(starts) > 0:
                    if strand == '+':
                        orf = ORF(front, back, starts, strand, frame)
                    else:
                        orf = ORF(len(dna)-back, len(dna)-front, list(reversed([len(dna)-start for start in starts])), strand, frame)
                    self.orfs.append(orf)

    def mark_coding_orfs(self, realFeatures):
        orfsPos = {o.right:o for o in self.orfs if o.strand == '+'}
        orfsNeg = {o.left:o for o in self.orfs if o.strand == '-'}

        featuresPos = [f for f in realFeatures if f.strand == '+']
        featuresNeg = [f for f in realFeatures if f.strand == '-']

        for feature in featuresPos:
            if feature.right in orfsPos:
                orfsPos[feature.right].realStart = feature.left

        for feature in featuresNeg:
            if feature.left in orfsNeg:
                orfsNeg[feature.left].realStart = feature.right

    @property
    def dnaPos(self):
        return self._dnaPos
    @dnaPos.setter
    def dnaPos(self, dnaPos):
        self._dnaPos = dnaPos
        self._dnaNeg = dna_reverse_complement(dnaPos)

        a = self._dnaPos.count('a')
        c = self._dnaPos.count('c')
        g = self._dnaPos.count('g')
        t = self._dnaPos.count('t')
        self.gc = (g+c) / (a+c+g+t)

    @property
    def dnaNeg(self):
        return self._dnaNeg
    @dnaNeg.setter
    def dnaNeg(self, dnaNeg):
        self._dnaNeg = dnaNeg
        self._dnaPos = dna_reverse_complement(dnaNeg)

        a = self._dnaPos.count('a')
        c = self._dnaPos.count('c')
        g = self._dnaPos.count('g')
        t = self._dnaPos.count('t')
        self.gc = (g+c) / (a+c+g+t)

class Feature(object):
    """A section of a sequence with various properties."""

    def __init__(self, contig, left, right, strand, featureType, source, other=''):
        self.contig = contig
        self.left = left
        self.right = right
        self.strand = strand
        self.featureType = featureType
        self.source = source
        self.other = other

    def __str__(self):
        rv = ''

        if self.contig is not None:
            rv += ' contig=' + self.contig.__repr__()

        if self.left is not None:
            rv += ' left=' + self.left.__repr__()

        if self.right is not None:
            rv += ' right=' + self.right.__repr__()

        if self.strand is not None:
            rv += ' strand=' + self.strand.__repr__()

        if self.featureType is not None:
            rv += ' featureType=' + self.featureType.__repr__()

        if self.source is not None:
            rv += ' source=' + self.source.__repr__()

        if self.other is not None:
            rv += ' other=' + self.other.__repr__()

        return '<Feature' + rv + '>'

    def __repr__(self):
        return self.__str__()

    def encode_gff(self):
        return '\t'.join([
            self.contig,
            self.source,
            self.featureType,
            self.left,
            self.right,
            '.',
            self.strand,
            '0',
            self.other,
        ])

class ORF(object):
    """Specialized feature for open reading frames."""

    def __init__(self, left, right, starts, strand, frame):
        self.left = left
        self.right = right
        self.starts = starts
        self.strand = strand
        self.frame = frame
        self.scores = {}
        self.realStart = None

    def __str__(self):
        rv = ''

        if self.left is not None:
            rv += ' left=' + self.left.__repr__()

        if self.right is not None:
            rv += ' right=' + self.right.__repr__()

        if self.starts is not None:
            rv += ' starts=' + self.starts.__repr__()

        if self.strand is not None:
            rv += ' strand=' + self.strand.__repr__()

        if self.frame is not None:
            rv += ' frame=' + self.frame.__repr__()

        return '<ORF' + rv + '>'

    def __repr__(self):
        return self.__str__()

    @property
    def score(self):
        if self.scores is {}:
            return 0

        return max(self.scores.values())

def read_genomes_from_list(genomeDir, genomeList):
    """From a list of genome ID's, read the genomes into Genome objects."""
    genomes = []

    with open(genomeList) as f:
        for line in f.readlines():
            genomeId = line.strip()
            g = Genome(gid=genomeId, source=genomeDir+'/'+genomeId)
            genomes.append(g)

    return genomes

def read_fasta(fname):
    """Read contigs from file fname as DNA strings."""
    name2seq = {}

    with open(fname) as f:
        currentName = ''

        for line in f:
            if line.startswith('>'):
                #if line.find(' ') != -1:
                #    currentName = line[1:line.find(' ')]
                #else:
                #    currentName = line[1:]
                currentName = line[1:].split()[0]
                name2seq[currentName] = []
            else:
                name2seq[currentName].append(line.strip().lower())

        for name in name2seq.keys():
            name2seq[name] = ''.join(name2seq[name])

    return name2seq

def read_gff(fname):
    """Read features from file fname."""
    contig2features = {}

    with open(fname) as f:
        for line in f:
            if '#' in line:
                line = line[:line.find('#')]

            line = line.strip().split('\t')

            if len(line) < 3:
                continue

            contig = line[0]
            if contig.startswith('accn|'):
                contig = contig[5:]

            if contig not in contig2features:
                contig2features[contig] = []

            source = line[1]
            featureType = line[2]
            start = int(line[3])-1
            stop = int(line[4])
            strand = line[6]
            other = line[8]
            # product = dict(p.split('=') for p in line[8].split(';')).get('product', None)

            # contig2features[contig].append((start, stop, strand, featureType, product))
            contig2features[contig].append(Feature(contig, start, stop, strand, featureType, source, other))

    return contig2features

def read_gto(fname):
    with open(fname) as f:
        data = json.load(f)

    g = Genome(gid=data['id'])
    cid2c = {}

    for contig in data['contigs']:
        c = Contig(cid=contig['id'], genome=g)
        c.dnaPos = contig['dna']

        cid2c[c.cid] = c
        g.contigs.append(c)

    for feature in data['features']:
        loc = feature['location'][0]
        f = Feature(contig=loc[0], left=int(loc[1])-1, right=int(loc[1])-1+int(loc[3]), strand=loc[2], featureType=feature['type'], source='PATRIC')
        if len(feature['location']) > 1:
            print('hmm', feature['location'], f)
        cid2c[loc[0]].features.append(f)

    return g

def save_gff(fname, features):
    """Save features to file fname."""
    with open(fname, 'w') as f:
        print('##gff-version 3', file=f)

        for feature in features:
            print(feature.encode_gff(), file=f)

def save_gto(fname, genome):
    data = {}

    data['id'] = genome.gid
    data['contigs'] = []
    data['features'] = []

    for contig in genome.contigs:
        cD = {}
        cD['id'] = contig.cid
        cD['dna'] = contig.dnaPos
        data['contigs'].append(cD)

        for feature in contig.features:
            fD = {}
            fD['type'] = feature.featureType
            fD['location'] = [[feature.contig, feature.left+1, feature.strand, feature.right-feature.left]]
            data['features'].append(fD)

    with open(fname, 'w') as f:
        json.dump(data, f, indent=3)

def save_gpt(fname, features):
    """Save features to file fname."""
    with open(fname, 'w') as f:
        print('contig.id\tleft\tright\tconfidence\tstrand\ttype', file=f)

        for feature in features:
            if feature.source != 'RMB':
                continue

            for k, v in [i.split('=') for i in feature.other.split(';')]:
                if k == 'rmbscore':
                    score = float(v)
                    break
            else:
                score = 0

            print('\t'.join((feature.contig, str(feature.left), str(feature.right), '%0.6f'%score, feature.strand, feature.featureType)), file=f)



class Visualizer(object):
    """Convert sequence and feature objects into images."""

    def __init__(self, length):
        print('Visualizer has not been updated to the new ORF format')

        self.length = length
        self.dna = ''
        self.seqlen = 0

        self.img_spacer = np.ones((1, length, 3), dtype=np.uint8) * 255

        self.img_dna_p = np.ones((4, length, 3), dtype=np.uint8) * 255
        self.img_dna_n = np.ones((4, length, 3), dtype=np.uint8) * 255

        self.img_orf_frame_1 = np.ones((4, length, 3), dtype=np.uint8) * 255
        self.img_orf_frame_2 = np.ones((4, length, 3), dtype=np.uint8) * 255
        self.img_orf_frame_3 = np.ones((4, length, 3), dtype=np.uint8) * 255
        self.img_orf_frame_m1 = np.ones((4, length, 3), dtype=np.uint8) * 255
        self.img_orf_frame_m2 = np.ones((4, length, 3), dtype=np.uint8) * 255
        self.img_orf_frame_m3 = np.ones((4, length, 3), dtype=np.uint8) * 255

        self.img_start_frame_1 = np.ones((14, length, 3), dtype=np.uint8) * 255
        self.img_start_frame_2 = np.ones((14, length, 3), dtype=np.uint8) * 255
        self.img_start_frame_3 = np.ones((14, length, 3), dtype=np.uint8) * 255
        self.img_start_frame_m1 = np.ones((14, length, 3), dtype=np.uint8) * 255
        self.img_start_frame_m2 = np.ones((14, length, 3), dtype=np.uint8) * 255
        self.img_start_frame_m3 = np.ones((14, length, 3), dtype=np.uint8) * 255

        self.img_feature_frame_1 = np.ones((8, length, 3), dtype=np.uint8) * 255
        self.img_feature_frame_2 = np.ones((8, length, 3), dtype=np.uint8) * 255
        self.img_feature_frame_3 = np.ones((8, length, 3), dtype=np.uint8) * 255
        self.img_feature_frame_m1 = np.ones((8, length, 3), dtype=np.uint8) * 255
        self.img_feature_frame_m2 = np.ones((8, length, 3), dtype=np.uint8) * 255
        self.img_feature_frame_m3 = np.ones((8, length, 3), dtype=np.uint8) * 255

        self.img_pfeature_frame_1 = np.ones((8, length, 3), dtype=np.uint8) * 255
        self.img_pfeature_frame_2 = np.ones((8, length, 3), dtype=np.uint8) * 255
        self.img_pfeature_frame_3 = np.ones((8, length, 3), dtype=np.uint8) * 255
        self.img_pfeature_frame_m1 = np.ones((8, length, 3), dtype=np.uint8) * 255
        self.img_pfeature_frame_m2 = np.ones((8, length, 3), dtype=np.uint8) * 255
        self.img_pfeature_frame_m3 = np.ones((8, length, 3), dtype=np.uint8) * 255

        self.img_rna_profile = np.ones((64, length, 3), dtype=np.uint8) * 255

        self.img_dna_scale = np.ones((16, length, 3), dtype=np.uint8) * 255

    def add_dna(self, dna):
        self.dna = dna
        self.seqlen = len(dna)

        if self.length == len(dna):
            self.paint_dna(self.img_dna_p, dna)
            self.paint_dna(self.img_dna_n, dna_reverse_complement(dna)[::-1])

    def add_rna(self, rnaProfile):
        convertedProfile = np.clip(rnaProfile, 0, 100)/100

        height = len(self.img_rna_profile)

        for i in range(self.length):
            p1 = i*self.seqlen//self.length
            p2 = (i+1)*self.seqlen//self.length

            if convertedProfile[p1:p2].size == 0:
                s = 0
            else:
                s = np.max(convertedProfile[p1:p2])

            s = int((1-s) * height)

            self.img_rna_profile[s:height, i] = (231, 213, 188)

    def add_orfs(self, orfs):
        for orf in orfs:
            if orf.strand == '+':
                front = orf.front
                back = orf.back - 1

                if orf.frame == 0:
                    img_orf = self.img_orf_frame_1
                    img_start = self.img_start_frame_1
                if orf.frame == 1:
                    img_orf = self.img_orf_frame_2
                    img_start = self.img_start_frame_2
                if orf.frame == 2:
                    img_orf = self.img_orf_frame_3
                    img_start = self.img_start_frame_3
            else:
                back = self.seqlen - orf.back
                front = self.seqlen - orf.front - 1

                if orf.frame == 0:
                    img_orf = self.img_orf_frame_m1
                    img_start = self.img_start_frame_m1
                if orf.frame == 1:
                    img_orf = self.img_orf_frame_m2
                    img_start = self.img_start_frame_m2
                if orf.frame == 2:
                    img_orf = self.img_orf_frame_m3
                    img_start = self.img_start_frame_m3

            front = self.length * front // self.seqlen
            back = self.length * back // self.seqlen

            self.paint_arrow(img_orf, front, back, orf.strand, (240, 240, 240), (180, 180, 180))

            for start in orf.starts:
                if orf.strand == '+':
                    startPos = start
                else:
                    startPos = self.seqlen - start - 1
                startPos = self.length * startPos // self.seqlen
                # self.paint_start(img_orf, startPos, 1)
                self.paint_start(img_start, startPos, orf.scores.get(start, 0))

    def add_feature(self, feature, color, track):
        if track == 0:
            if feature.strand == '+':
                front = feature.front
                back = feature.back - 1
                frame = front%3

                if frame == 0:
                    img_feature = self.img_feature_frame_1
                if frame == 1:
                    img_feature = self.img_feature_frame_2
                if frame == 2:
                    img_feature = self.img_feature_frame_3
            else:
                front = feature.back - 1
                back = feature.front
                frame = (back+1)%3

                if frame == 0:
                    img_feature = self.img_feature_frame_m3
                if frame == 1:
                    img_feature = self.img_feature_frame_m2
                if frame == 2:
                    img_feature = self.img_feature_frame_m1
        else:
            if feature.strand == '+':
                front = feature.front
                back = feature.back - 1
                frame = front%3

                if frame == 0:
                    img_feature = self.img_pfeature_frame_1
                if frame == 1:
                    img_feature = self.img_pfeature_frame_2
                if frame == 2:
                    img_feature = self.img_pfeature_frame_3
            else:
                front = feature.back - 1
                back = feature.front
                frame = (back+1)%3

                if frame == 0:
                    img_feature = self.img_pfeature_frame_m3
                if frame == 1:
                    img_feature = self.img_pfeature_frame_m2
                if frame == 2:
                    img_feature = self.img_pfeature_frame_m1

        front = self.length * front // self.seqlen
        back = self.length * back // self.seqlen

        self.paint_arrow(img_feature, front, back, feature.strand, color, (0, 0, 0))

    def add_scale(self):
        spacing = 1000

        numTicks = self.seqlen//spacing
        for i in range(1, numTicks):
            pixPos = self.length*(i*spacing)//self.seqlen
            self.img_dna_scale[0:4, pixPos] = (0, 0, 0)
            cv2.putText(self.img_dna_scale, str(i*spacing), (pixPos-10, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    def save(self, filename):
        self.add_scale()

        # labels = self.paint_axis_label(dna_height, frame_height, rna_profile_height, scale_height)

        img = np.concatenate((
            self.img_rna_profile,
            self.img_start_frame_3,
            self.img_orf_frame_3,
            self.img_feature_frame_3,
            self.img_pfeature_frame_3,
            self.img_start_frame_2,
            self.img_orf_frame_2,
            self.img_feature_frame_2,
            self.img_pfeature_frame_2,
            self.img_start_frame_1,
            self.img_orf_frame_1,
            self.img_feature_frame_1,
            self.img_pfeature_frame_1,
            self.img_dna_p,
            self.img_dna_n,
            self.img_start_frame_m1,
            self.img_orf_frame_m1,
            self.img_feature_frame_m1,
            self.img_pfeature_frame_m1,
            self.img_start_frame_m2,
            self.img_orf_frame_m2,
            self.img_feature_frame_m2,
            self.img_pfeature_frame_m2,
            self.img_start_frame_m3,
            self.img_orf_frame_m3,
            self.img_feature_frame_m3,
            self.img_pfeature_frame_m3,
            self.img_dna_scale,
        ), axis=0)

        # img = np.concatenate((
        #     labels,
        #     img,
        # ), axis=1)

        cv2.imwrite(filename, img)

    def paint_arrow(self, img, front, back, strand, color, color2=(0, 0, 0)):
        top = 0
        bot = len(img)-1
        mid = len(img)//2-1

        if strand == '+':
            if back-front < mid:
                front = back-mid

            contours = np.array([
                [front, top],
                [back-mid, top],
                [back, top+mid],
                [back, bot-mid],
                [back-mid, bot],
                [front, bot]
            ])
        else:
            # if front-back < mid:
            #     back = front-mid

            contours = np.array([
                [front, top],
                [back+mid, top],
                [back, top+mid],
                [back, bot-mid],
                [back+mid, bot],
                [front, bot]
            ])

        cv2.drawContours(img, contours=[contours], contourIdx=-1, color=color, thickness=cv2.FILLED)
        cv2.drawContours(img, contours=[contours], contourIdx=-1, color=color2, thickness=1)

    def paint_start(self, img, start, score):
        top = len(img) - (int(len(img)*score)) - 1
        bot = len(img) - 1

        contours = np.array([
            [start, top],
            [start, bot]
        ])

        cv2.drawContours(img, contours=[contours], contourIdx=-1, color=(0, 0, 0), thickness=1)

    def paint_dna(self, img, dna):
        colors = {
            'a': ( 32, 189,  28),
            'c': (251,  80,  73),
            'g': ( 41, 186, 212),
            't': ( 27,  13, 252),
        }

        for i in range(len(dna)):
            img[:, i] = colors[dna[i]]

    def paint_axis_label(self, dna_height, frame_height, rna_profile_height, scale_height):
        total_height = dna_height*2 + frame_height*6 + rna_profile_height + scale_height
        img = np.ones((total_height, 50, 3), dtype=np.uint8) * 255

        cv2.putText(img, 'RNA', (20, rna_profile_height//2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        cv2.putText(img, '+3', (25, rna_profile_height + frame_height*0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.putText(img, '+2', (25, rna_profile_height + frame_height*1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.putText(img, '+1', (25, rna_profile_height + frame_height*2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        cv2.putText(img, '-1', (25, rna_profile_height + frame_height*3 + dna_height*2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.putText(img, '-2', (25, rna_profile_height + frame_height*4 + dna_height*2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.putText(img, '-3', (25, rna_profile_height + frame_height*5 + dna_height*2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        return img



def score_regions(samples, scoreModel):
    return scoreModel.predict(samples[:, :-2], batch_size=4096)

def calc_loc_preds(dna, starts, stops, startL, startR, stopL, stopR, codingSize, startModel, stopModel, codingModel):
    startLocs = []
    startSamples = []
    startCodingSamples = []
    stopLocs = []
    stopSamples = []
    stopCodingSamples = []

    dna = np.array(to_sparse_categorical(list(dna), encoder=dnaEncoder))

    # Find all possible starts
    for location in starts:
        sample = dna[location-startL:location+3+startR]
        codingSample = dna[location+3+startR:location+3+startR+codingSize]

        # Sample might have been at the edge of a contig
        if len(sample) != startL+3+startR:
            continue
        if len(codingSample) != codingSize:
            continue

        startLocs.append(location)
        startSamples.append(sample)
        startCodingSamples.append(codingSample)

    # Find all possible stops
    for location in stops:
        sample = dna[location-stopL:location+3+stopR]
        codingSample = dna[location-stopL-codingSize:location-stopL]

        # Sample might have been at the edge of a contig
        if len(sample) != stopL+3+stopR:
            continue
        if len(codingSample) != codingSize:
            continue

        stopLocs.append(location)
        stopSamples.append(sample)
        stopCodingSamples.append(codingSample)

    startSamples = np.array(startSamples)
    stopSamples = np.array(stopSamples)
    startCodingSamples = np.array(startCodingSamples)
    stopCodingSamples = np.array(stopCodingSamples)

    # startSamples = np.array([to_sparse_categorical(list(s), encoder=dnaEncoder) for s in startSamples])
    # stopSamples = np.array([to_sparse_categorical(list(s), encoder=dnaEncoder) for s in stopSamples])
    # startCodingSamples = np.array([to_sparse_categorical(list(s), encoder=dnaEncoder) for s in startCodingSamples])
    # stopCodingSamples = np.array([to_sparse_categorical(list(s), encoder=dnaEncoder) for s in stopCodingSamples])

    if startSamples.size == 0 or stopSamples.size == 0 or startCodingSamples.size == 0 or stopCodingSamples.size == 0:
        return {}, {}, {}, {}

    # Process the start and stop dna segments
    startPreds = startModel.predict(startSamples, batch_size=4096)
    stopPreds = stopModel.predict(stopSamples, batch_size=4096)
    startCodingPreds = codingModel.predict(startCodingSamples, batch_size=4096)
    stopCodingPreds = codingModel.predict(stopCodingSamples, batch_size=4096)

    startLoc2Pred = dict(zip(startLocs, startPreds))
    stopLoc2Pred = dict(zip(stopLocs, stopPreds))
    startLoc2CodingPred = dict(zip(startLocs, startCodingPreds))
    stopLoc2CodingPred = dict(zip(stopLocs, stopCodingPreds))

    return startLoc2Pred, stopLoc2Pred, startLoc2CodingPred, stopLoc2CodingPred

def combine_orf_preds(orf, startLoc2Pred, stopLoc2Pred, startLoc2CodingPred, stopLoc2CodingPred):
    print('combine_orf_preds has not been updated to the new ORF format')

    if orf.back-3 not in stopLoc2Pred:
        return []

    i = 0
    genes = []
    for start in orf.starts:
        if start not in startLoc2Pred:
            continue

        genes.append([
            startLoc2Pred[start][0],
            *startLoc2CodingPred[start].flatten(),
            *stopLoc2CodingPred[orf.back-3].flatten(),
            stopLoc2Pred[orf.back-3][0],
            orf.back-start,
            start,
            orf.back,
        ])

        i += 1

    return genes

def score_orfs(contig, cutoff, startL, startR, stopL, stopR, codingSize, startModel, stopModel, codingModel, scoreModel):
    startLoc2Pred, stopLoc2Pred, startLoc2CodingPred, stopLoc2CodingPred = calc_loc_preds(
        contig.dnaPos,
        [start for orf in contig.orfsPos for start in orf.starts],
        [orf.back-3 for orf in contig.orfsPos],
        startL,
        startR,
        stopL,
        stopR,
        codingSize,
        startModel,
        stopModel,
        codingModel,
    )

    for orf in contig.orfsPos:
        samples = combine_orf_preds(
            orf,
            startLoc2Pred,
            stopLoc2Pred,
            startLoc2CodingPred,
            stopLoc2CodingPred,
        )

        if len(samples) == 0:
            continue

        samples = np.array(samples)
        scores = score_regions(samples, scoreModel)

        for i in range(len(samples)):
            orf.scores[int(samples[i][-2])] = float(scores[i])

        if scores.max() >= cutoff:
            # for sample in samples:
            best = samples[scores.argmax()]

            front = int(best[-2])
            back = int(best[-1])
            contig.features.append(Feature(contig.cid, front, back, '+', 'CDS', 'RMB', other='rmbscore=%.6f'%(orf.scores[front])))

            # print(samples)
            # print(scores)
            # print([start for start in orf.starts][:10])
            # print(orf.scores)
            # print(scores.argmax())
            # print(scores.max())
            # print(best)

    # startLoc2Pred, stopLoc2Pred, startLoc2CodingPred, stopLoc2CodingPred = calc_loc_preds(
    #     contig.dnaNeg,
    #     [start for orf in contig.orfsNeg for start in orf.starts],
    #     [orf.back-3 for orf in contig.orfsNeg],
    #     startL,
    #     startR,
    #     stopL,
    #     stopR,
    #     codingSize,
    #     startModel,
    #     stopModel,
    #     codingModel,
    # )

    # for orf in contig.orfsNeg:
    #     samples = combine_orf_preds(
    #         orf,
    #         startLoc2Pred,
    #         stopLoc2Pred,
    #         startLoc2CodingPred,
    #         stopLoc2CodingPred,
    #     )

    #     if len(samples) == 0:
    #         continue

    #     samples = np.array(samples)
    #     scores = score_regions(samples, scoreModel)

    #     best = samples[scores.argmax()]
    #     gene = [len(contig.dnaNeg)-best[-1], len(contig.dnaNeg)-best[-2], scores.max(), '-']
    #     # genes.append(gene)

    #     #for sample in samples:
    #     #    gene = [len(contig.dnaNeg)-sample[-1], len(contig.dnaNeg)-sample[-2], scores.max(), '-']
    #     #    genes.append(gene)

    #     front = int(len(contig.dnaNeg)-best[-1])
    #     back = int(len(contig.dnaNeg)-best[-2])
    #     strand = '-'
    #     featureType = 'CDS'
    #     source = 'RMB'
    #     f = Feature(contig, front, back, strand, featureType, source, other='')
