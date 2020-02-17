import pytest
import utils

def test_Genome():
    genome = utils.Genome(gid='1000562.3', source='genomes_proks/1000562.3')
    
    assert genome.__repr__() == "<Genome name='1000562.3' source='genomes_proks/1000562.3'; 105 contigs; 1,659,203 bp; 39.6% gc>"
    assert genome.contigs[0].__repr__() == "<Contig name='JSAP01000001'; 21,134 bp; 41.8% gc; 24 features; 0 orfs>"

    contig = genome.contigs[0]
    contig.find_orfs()
    contig.mark_coding_orfs()
    print(contig)

if __name__ == '__main__':
    test_Genome()
