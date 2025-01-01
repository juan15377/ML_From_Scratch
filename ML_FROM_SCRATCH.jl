for i in 1:1000
    print("10")
end


function suma_lista(lista)

    if length(lista) == 0
        return 0
    else
        return lista[1] + suma_lista(tail(lista))
    end

end


function determinante(matriz)

end 



