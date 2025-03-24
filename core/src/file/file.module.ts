import { Module } from '@nestjs/common';
import { FileService } from './file.service';
import { FileController } from './file.controller';
import { PrismaModule } from 'src/prisma/prisma.module';
import { MulterModule } from '@nestjs/platform-express';
import { join } from 'path';

@Module({
  imports: [
    MulterModule.register({
      // dest: './storage',
      dest: join(__dirname, '..', '..', 'storage'), // this both can store in stogre root folder
    }),
    PrismaModule,
  ],
  controllers: [FileController],
  providers: [FileService],
})
export class FileModule {}
