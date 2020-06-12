function [fea] = convert_csv_to_mat_feat(filename)
    
    % calculating the number of rows in the file
    fidr = fopen(filename, 'r'); 
    
    N = 784;
    cac = textscan(fidr,[repmat('%f',[1,N]), '%s'],'CollectOutput',1,'Delimiter',','  ... 
        , 'Headerlines',1, 'Headercolumns',0 );
   
    fclose(fidr); 
    [rows,cols] = cellfun(@size,cac,'uni',false);
    rs = rows{1,1};
    cs = cols{1,1};
        
    fea = cac{:,1:cs};
    %gnd = cac{:,2};

end
