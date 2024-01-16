import numpy as np
import pandas as pd
import scipy
import scipy.sparse
class ChipSeq_data_convert:
    def __init__(self):
        self.start_position_to_gene={}
        self.start_position=[]

        self.promoter_region_start=[]
        self.promoter_region_end=[]
        self.promoter_chrom=[]

        self.single_cell_exp_set=[]

        self.peak_set=[]#TF to peak position chr_start_end, split by ,
        self.positive_pair=[]#TF to gene, split by ,

        #########
        self.expr = None
        self.geneNames = None
        self.geneIDs = None #for lung data
        self.corr_matrix = None
        self.filter_by_corr = None
        self.flag_filter=True
        self.flag_remove_corr_NA=False
        self.flag_negative_also_large_corr=False
        self.adj_matrix = None

        self.TF_list = []
        self.rows_for_TF = []
        self.filtered_TF_rows = []
        self.filtered_candidate_genes_rows = []



    def get_index_by_geneName(self,genei):
        index=np.where(self.geneNames==genei)
        if len(index[0]) > 0:
            return index[0]
        else:
            return None
    def get_index_by_geneID(self,genei):
        index=np.where(self.geneIDs==genei)
        if len(index[0]) > 0:
            return index[0]
        else:
            return None

    def get_exp_by_geneName(self,genei):
        index = np.where(self.geneNames == genei)
        from scipy import sparse
        if len(index[0]) > 0:
            row = index[0]
            if isinstance(self.expr,sparse.csr_matrix):#isinstance() 函数来判断一个对象是否是一个已知的类型
                genei_exp = np.asarray(self.expr.tocsr()[row, :].todense())#todense（）将矩阵显示出来
            else:
                genei_exp = self.expr[row, :]
            return genei_exp.ravel()
        else:
            return None

    def read_positive_pairs_and_filter(self,filename, reverse_corr_filter=False,col0_get_by_name=True, col1_get_by_name=True):
        positive_pairs=pd.read_csv(filename,header=None)
        #change the positive pairs to the index in self.geneNames.
        V = []
        I = []
        J = []
        print("positive_pairs",positive_pairs)
        print("self.geneNames", self.geneNames)
        print("self.geneIDs", self.geneIDs)
        for i in range(0,positive_pairs.shape[0]):
            if col0_get_by_name:
                indexA=self.get_index_by_geneName(positive_pairs.iloc[i,0])
            else:
                indexA = self.get_index_by_geneID(positive_pairs.iloc[i, 0])
            if col1_get_by_name:
                indexB=self.get_index_by_geneName(positive_pairs.iloc[i,1])
            else:
                indexB = self.get_index_by_geneID(positive_pairs.iloc[i, 1])

            if indexA is not None and indexB is not None:
                V.append(1)
                I.append(indexA)
                J.append(indexB)
                print("find gene name in positive pair", positive_pairs.iloc[i, 0], positive_pairs.iloc[i, 1])

            else:
                if col0_get_by_name:
                    print("col0 by name")
                    indexA = self.get_index_by_geneName(positive_pairs.iloc[i, 0].lower())

                else:
                    print("col0 by ID")
                    indexA = self.get_index_by_geneID(positive_pairs.iloc[i, 0].lower())
                if col1_get_by_name:
                    print("col1 by name")
                    indexB = self.get_index_by_geneName(positive_pairs.iloc[i, 1].lower())
                else:
                    print("col1 by ID")
                    indexB = self.get_index_by_geneID(positive_pairs.iloc[i, 1].lower())

                if indexA is not None and indexB is not None:
                    V.append(1)
                    I.append(indexA)
                    J.append(indexB)
                    print("find gene name in positive pair", positive_pairs.iloc[i, 0].lower(), positive_pairs.iloc[i, 1].lower())
                else:
                    print("can not find gene name in positive pair", positive_pairs.iloc[i,0].lower(), positive_pairs.iloc[i,1].lower())

        V = np.asarray(V)
        V = V.ravel()
        I = np.asarray(I)
        I = I.ravel()
        J = np.asarray(J)
        J = J.ravel()
        print("V", V)
        print("I", I)
        print("J", J)
        if self.flag_negative_also_large_corr:
            V = np.ones((len(I),))

        self.adj_matrix= scipy.sparse.csr_matrix((V, (I, J)), shape=(len(self.geneNames), len(self.geneNames)))#all 0, if positive pairs set as 1, an sparse matrix
        self.adj_matrix=self.adj_matrix.toarray()

        print("corr_matrix",self.corr_matrix)
        print("adj_matrix",self.adj_matrix)

        if self.flag_filter:
            if self.flag_remove_corr_NA:
                self.adj_matrix = np.where(np.isnan(self.corr_matrix), 0,self.adj_matrix)
            if self.filter_by_corr:
                if self.flag_negative_also_large_corr:
                    self.adj_matrix = np.where(np.abs(self.corr_matrix) > self.corr_cutoff, self.adj_matrix, 2)

                else:

                    if reverse_corr_filter:
                        self.adj_matrix=np.where(np.abs(self.corr_matrix) > self.corr_cutoff, 0, self.adj_matrix)
                    else:
                        self.adj_matrix = np.where(np.abs(self.corr_matrix) > self.corr_cutoff, self.adj_matrix, 0)
        return self.adj_matrix
    def output_positive_pairs_according_to_adj_matrix(self,out_filename,lowerCase=False):
        if self.flag_negative_also_large_corr:
            result = np.where(self.adj_matrix == 1)
        else:
            result = np.where(self.adj_matrix!=0)
        print("result",result)
        rows=result[0]
        cols=result[1]
        print("rows",rows)
        geneA_names=self.geneNames[rows]
        geneB_names=self.geneNames[cols]
        print("geneA_names",geneA_names)
        print("geneB_names", geneB_names)
        if lowerCase:
            for i in range(0,len(geneA_names)):
                geneA_names[i]=geneA_names[i].lower()
                geneB_names[i]=geneB_names[i].lower()
        df=pd.DataFrame(geneA_names,geneB_names)
        df.to_csv(out_filename,sep="\t",header=False)

    def output_neg_pairs_according_to_adj_matrix(self,out_filename,lowerCase=False):

        result = np.where(self.adj_matrix==0)
        print("result",result)
        rows=result[0]
        cols=result[1]
        print("rows",rows)
        geneA_names=self.geneNames[rows]
        geneB_names=self.geneNames[cols]
        print("geneA_names",geneA_names)
        print("geneB_names", geneB_names)
        if lowerCase:
            for i in range(0,len(geneA_names)):
                geneA_names[i]=geneA_names[i].lower()
                geneB_names[i]=geneB_names[i].lower()
        df=pd.DataFrame(geneA_names,geneB_names)
        df.to_csv(out_filename,sep="\t",header=False)

    def output_training_pairs_according_to_adj_matrix(self,out_filename,lowerCase=True, random_training_pairs=False):
        # one positive pair 1, one positive pair in other direction 2, one negative pair 0
        # since we have get genes, we can generate an adjacency matrix for TF to all genes, and sample pairs of TF to random genes
        # all genes in lower case.
        from random import shuffle
        #get_positive:
        if self.flag_negative_also_large_corr:
            result = np.where(self.adj_matrix == 1)
        else:
            result = np.where(self.adj_matrix != 0)
        positive_rows = result[0]
        positive_cols = result[1]
        print("positive_rows",positive_rows)
        print("len(positive_rows)",len(positive_rows))
        if random_training_pairs:
            indexs = np.arange(len(positive_rows))
            shuffle(indexs)
            positive_rows = positive_rows[indexs]
            positive_cols = positive_cols[indexs]

        negative_rows = []
        negative_cols = []
        negative_rows = np.asarray(negative_rows, dtype=int)
        negative_cols = np.asarray(negative_cols, dtype=int)

        reordered_positive_rows = []
        reordered_positive_cols = []

        result = np.where(self.adj_matrix == 0)
        negative_rows = result[0]
        negative_cols = result[1]
        print("negative_rows", negative_rows)
        print("len(negative_rows)", len(negative_rows))

        geneA_names_positive = self.geneNames[positive_rows]
        geneB_names_positive = self.geneNames[positive_cols]
        geneA_names_negative = self.geneNames[negative_rows]
        geneB_names_negative = self.geneNames[negative_cols]
        # label_negative = np.zeros(len(geneA_names_negative, 1))
        print("len(geneA_names_negative)", len(geneA_names_negative))
        print("len(geneB_names_negative)", len(geneA_names_negative))
        print("len(geneA_names_positive)", len(geneA_names_positive))
        print("len(geneB_names_positive)", len(geneB_names_positive))

        out_string_list = []
        out_string_list1=[]
        for i in range(0, len(geneA_names_positive)):
            if lowerCase:
                geneA_names_positive[i] = geneA_names_positive[i].lower()
                geneB_names_positive[i] = geneB_names_positive[i].lower()
                #geneA_names_negative[i] = geneA_names_negative[i].lower()
                #geneB_names_negative[i] = geneB_names_negative[i].lower()
            positive_pair = str(geneA_names_positive[i]) + '\t' + str(geneB_names_positive[i]) + '\t' + str(1)
            #negative_pair = str(geneA_names_negative[i]) + '\t' + str(geneB_names_negative[i]) + '\t' + str(0)
            out_string_list.append(positive_pair)
            #out_string_list.append(negative_pair)
        for i in range(0, len(result[0])):
            if lowerCase:
                geneA_names_negative[i] = geneA_names_negative[i].lower()
                geneB_names_negative[i] = geneB_names_negative[i].lower()
            negative_pair = str(geneA_names_negative[i]) + '\t' + str(geneB_names_negative[i]) + '\t' + str(0)
            out_string_list1.append(negative_pair)
        out_string_list=out_string_list+out_string_list1
        out_string_list = np.asarray(out_string_list)
        np.savetxt('00000-adj_matrix.csv', self.adj_matrix, fmt='%s', delimiter=',')
        np.savetxt(out_filename, out_string_list, fmt="%s", delimiter='\n')


    def output_training_pairs_according_to_adj_matrix1(self,out_filename,lowerCase=True, random_training_pairs=False):
        # one positive pair 1, one positive pair in other direction 2, one negative pair 0
        # since we have get genes, we can generate an adjacency matrix for TF to all genes, and sample pairs of TF to random genes
        # all genes in lower case.
        from random import shuffle
        #get_positive:
        if self.flag_negative_also_large_corr:
            result = np.where(self.adj_matrix == 1)
        else:
            result = np.where(self.adj_matrix != 0)
        positive_rows = result[0]
        positive_cols = result[1]
        print("positive_rows",positive_rows)
        print("len(positive_rows)",len(positive_rows))
        if random_training_pairs:
            indexs = np.arange(len(positive_rows))
            shuffle(indexs)
            positive_rows = positive_rows[indexs]
            positive_cols = positive_cols[indexs]

        negative_rows = []
        negative_cols = []
        negative_rows = np.asarray(negative_rows, dtype=int)
        negative_cols = np.asarray(negative_cols, dtype=int)

        reordered_positive_rows = []
        reordered_positive_cols = []

        result = np.where(self.adj_matrix == 0)
        negative_rows = result[0]
        negative_cols = result[1]
        print("negative_rows", negative_rows)
        print("len(negative_rows)", len(negative_rows))

        geneA_names_positive = self.geneNames[positive_rows]
        geneB_names_positive = self.geneNames[positive_cols]
        geneA_names_negative = self.geneNames[negative_rows]
        geneB_names_negative = self.geneNames[negative_cols]
        # label_negative = np.zeros(len(geneA_names_negative, 1))
        print("len(geneA_names_negative)", len(geneA_names_negative))
        print("len(geneB_names_negative)", len(geneA_names_negative))
        print("len(geneA_names_positive)", len(geneA_names_positive))
        print("len(geneB_names_positive)", len(geneB_names_positive))

        out_string_list = []
        out_string_list1=[]
        for i in range(0, len(geneA_names_positive)):
            if lowerCase:
                geneA_names_positive[i] = geneA_names_positive[i].lower()
                geneB_names_positive[i] = geneB_names_positive[i].lower()
                #geneA_names_negative[i] = geneA_names_negative[i].lower()
                #geneB_names_negative[i] = geneB_names_negative[i].lower()
            positive_pair = str(geneA_names_positive[i]) + ',' + str(geneB_names_positive[i]) + ',' + str(1)
            #negative_pair = str(geneA_names_negative[i]) + '\t' + str(geneB_names_negative[i]) + '\t' + str(0)
            out_string_list.append(positive_pair)
            #out_string_list.append(negative_pair)
        for i in range(0, len(result[0])):
            if lowerCase:
                geneA_names_negative[i] = geneA_names_negative[i].lower()
                geneB_names_negative[i] = geneB_names_negative[i].lower()
            negative_pair = str(geneA_names_negative[i]) + ',' + str(geneB_names_negative[i]) + ',' + str(0)
            out_string_list1.append(negative_pair)
        out_string_list=out_string_list+out_string_list1
        out_string_list = np.asarray(out_string_list)
        np.savetxt(out_filename, out_string_list, fmt="%s", delimiter=',')


    def output_geneName_to_geneName_map(self,out_filename,lowerCase=True):
        if lowerCase:
            for i in range(0,len(self.geneNames)):
                self.geneNames[i] = self.geneNames[i].lower()
        if self.geneIDs is None:
            df=pd.DataFrame(self.geneNames,self.geneNames)
            df.to_csv(out_filename,sep="\t",header=False)
        else:
            df = pd.DataFrame({'geneName':self.geneNames, 'geneID':self.geneIDs})
            df.to_csv(out_filename, sep="\t", header=False,index=False)

    def get_geneNames_and_geneIDs_lower(self):
        if self.geneNames is not None:
            for i in range(0, len(self.geneNames)):
               self.geneNames[i] = self.geneNames[i].lower()

        if self.geneIDs is not None:
            for i in range(0, len(self.geneIDs)):
                self.geneIDs[i] = self.geneIDs[i].lower()

    def load_single_cell_type_expr(self, expr_file):
        if expr_file.endswith('.txt'):
            df = pd.read_table(expr_file, header='infer', index_col=0)
        else:
            df = pd.read_csv(expr_file,header='infer',index_col=0)

        self.expr = df.values
        print("expr.shape", self.expr.shape)

        self.geneNames = df.index
        self.geneNames = np.asarray(self.geneNames)
        self.geneNames = self.geneNames.ravel()
       # for i in range(0, len(self.geneNames)):
          #  self.geneNames[i] = self.geneNames[i].lower()

    def work_filter_positive_pair_single_cell_type(self, positive_pair_file, expr_file, label):
        self.filter_by_corr = False
        self.flag_remove_corr_NA = False
        self.flag_filter = False
        self.corr_cutoff = 0.1
        self.flag_negative_also_large_corr = False
        if self.flag_filter:
            self.load_single_cell_type_expr(expr_file)

            self.corr_matrix = np.corrcoef(self.expr)#np.corrcoef返回皮尔逊乘积距相关系数
            print("shape corr_matrix", self.corr_matrix.shape)
        else:
            self.load_single_cell_type_expr(expr_file)
        # read positive pair
        # filter positive pairs
        self.get_geneNames_and_geneIDs_lower()

        self.read_positive_pairs_and_filter(positive_pair_file, col0_get_by_name=True,
                                            col1_get_by_name=True)
        # output filtered positive pairs
        self.output_positive_pairs_according_to_adj_matrix("positive_pairs"+label+".txt")
        self.output_geneName_to_geneName_map(label+"_geneName_map.txt")
        # output training pairs
        self.output_training_pairs_according_to_adj_matrix("training_pairs"+label+".txt")
        self.output_training_pairs_according_to_adj_matrix1("training_pairs" + label + ".csv")
    def corr_only_nonzero(self,geneA_exp,geneB_exp):
        nonzero_indexA=np.where(geneA_exp!=0)
        nonzero_indexB=np.where(geneB_exp!=0)
        set1=set(nonzero_indexA[0])
        set2=set(nonzero_indexB[0])
        cells_for_corr=set1&set2
        if len(cells_for_corr)>0:
            print("cells for corr",cells_for_corr)
            cells_for_corr=list(cells_for_corr)
            vecA = geneA_exp[cells_for_corr]
            vecB = geneB_exp[cells_for_corr]
            corr = np.corrcoef(vecA,vecB)

            return corr[0,1]
        else:
            return 2


def main_single_cell_type_filter_positive_pair():
    tcs = ChipSeq_data_convert()
    tcs.work_filter_positive_pair_single_cell_type(positive_pair_file="D:/MEFFGRN/data/DREAM100/DREAM4_GoldStandard_InSilico_Size100_1.Csv",
                                                   expr_file="D:/MEFFGRN/data/Ecoil/cold_time_3_replice-1.csv", label="cold")
    #label="04621--04622"代表04622训练，04621独立测试
if __name__ == '__main__':
    main_single_cell_type_filter_positive_pair()
